import torch
import numpy as np
import pandas as pd
import datetime as dt
import random
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import os
import glob
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import torch.nn.init as weight_init
import scipy

device='cuda' if torch.cuda.is_available() else 'cpu'

class patient_encoder(nn.Module):

  """
  Takes constant demographic details and possibly labs and encodes into  z_0, h_0
  h_0 is the initial state for the RNN
  DO NOT WANT z_0=h_0
  """

  def __init__(self,constant_dim,z_dim=5,h_dim=64,hidden_dim=64,n_layers=3):
    
    super(patient_encoder,self).__init__()
    self.linear_input=nn.Linear(constant_dim,hidden_dim)
    self.h0=nn.Linear(hidden_dim,h_dim)
    self.z_0=nn.Linear(hidden_dim,z_dim)
    self.hidden_layers=nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ELU()) for i in range(n_layers)])
        
  def forward(self,x):
    """
    x=constants(+ may be initial_labs)
    """
    x=F.elu(self.linear_input(x)) #batch_size*hidden_dim

    for layer in self.hidden_layers:
      x=layer(x)
    
    h0=self.h0(x)
    z0=self.z_0(x)   
    
    # z_sigma=F.softplus(self.z_sigma(x))

    return z0,h0

class Encoder(nn.Module):
  """ Encodes x_{:t} for the variational distribution
      The job of the Encoder is to remember the past observations.
     adopted  from https://github.com/guxd/deepHMM/blob/master/modules.py#L163
  """

  def __init__(self,obs_dim,h_dim,a_dim,n_layers=1, dropout=0.0):
      
      super(Encoder,self).__init__()
      
      self.rnn=nn.GRU(obs_dim,h_dim,n_layers,batch_first=True)
      self.dropout=dropout  
      self.n_layers=n_layers
      self.hidden_dim=h_dim

      """
      THINK OF Init_h
      """
      
      # self.init_h = nn.Parameter(torch.randn(n_layers,1,
      #                                        h_dim), requires_grad=True)
      self.init_weights()

  def  init_weights(self):
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)

  def forward(self,full_obs,obs_lens,patient_demos,init_h=None, noise=False):

    """
    USING FULL_OBS for the moment, but later let's may be only use the 4d Obs
    
    obs: A mini batch of observations B*T*D
    obs_lens=observation lengths to pack pad sequences
    patient_demos: Batch of size B*pat_dim (=C)
    """
    batch_size, max_len, freq=full_obs.size()
    
    obs_lens=torch.LongTensor(obs_lens).to(device)
    obs_lens_sorted, indices = obs_lens.sort(descending=True)
    obs_sorted = full_obs.index_select(0, indices)  
    
    packed_obs=pack_padded_sequence(obs_sorted,obs_lens_sorted.data.tolist(),batch_first=True)
    # if init_h is None:
    #     init_h=self.init_h

    init_h=init_h.unsqueeze(0) #1*B*H
    
    
    hids, h_n = self.rnn(packed_obs, init_h) # hids: [B x T x H]  
                                                  # h_n: [num_layers*B*H)
    _, inv_indices = indices.sort()

    hids, lens = pad_packed_sequence(hids, batch_first=True)         
    hids = hids.index_select(0, inv_indices) 
    
            
    return hids

def mechanistic(Z):
  """
  Z should be torch.Tensor
  Z.shape =B*5
  Z[:,0]=HR,Z[:,1]=R_a, Z[:,2]=C_a, Z[:,3]=T , Z[:,4]=SV
  Obs Dim=4

  """
  obs=torch.zeros((Z.shape[0],4))
 
  
  # HR=Z[:,0]  #B
  # R=Z[:,1]
  # C=Z[:,2]
  # T=Z[:,3]
  # SV=Z[:,4]
  
  # obs[:,0]=HR
  # obs[:,1]=(SV/C)*(1/(1-torch.exp(-T/(R*C))))
  # obs[:,2]=obs[:,1]*torch.exp(-T/(R*C))
  # obs[:,3]=(SV/60)*HR*R

  obs[:,0]=Z[:,0]
  obs[:,1]=(Z[:,4]/Z[:,2])*(1/(1-torch.exp(-Z[:,3]/(Z[:,1]*Z[:,2]))))
  obs[:,2]=(torch.exp(-Z[:,3]/(Z[:,1]*Z[:,2])))*(Z[:,4]/Z[:,2])*(1/(1-torch.exp(-Z[:,3]/(Z[:,1]*Z[:,2]))))
  obs[:,3]=(Z[:,4]/60)*Z[:,0]*Z[:,1]


  return obs.to(device)

class Transition(nn.Module):
    """
    Deterministically Computes Z_t based on h_t,z_t_1,a_t
    using a MLP, where z is latent and a is the action
    Modified from https://pyro.ai/examples/dmm.html
    """
    def __init__(self, z_dim,a_dim,h_dim,constant_dim, hidden_dim=128,n_layers=8,mu_layers=4):
      super(Transition,self).__init__()
      
      input_dim=z_dim+a_dim+h_dim+constant_dim
      self.bn=nn.BatchNorm1d(input_dim)
      self.input_to_h=nn.Linear(input_dim, hidden_dim)
      
      self.hidden_layers=nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ELU()) for i in range(n_layers)])
      self.z=nn.Linear(hidden_dim,z_dim)
      

   
    def forward(self,z_prev,a_t,h_t,constants):
  
      # concatenate the latent z and actions along the frequency dimension
      input_=torch.cat([z_prev,a_t,h_t,constants],dim=1)
        #B*K*(D+T) let's call D+T=F
      
      # input_=self.bn(input_)
      input_=self.input_to_h(input_)
      
      for layer in self.hidden_layers:
        input_=layer(input_)     #B*K*H
      

      
      z=self.z(input_)
                           
      return z

class Emitter(nn.Module):
    """
    O_t based on Z_t
    """
    def __init__(self, z_dim,a_dim,h_dim,constant_dim, hidden_dim=128,n_layers=8):
      super(Transition,self).__init__()
      
      input_dim=z_dim+a_dim+h_dim+constant_dim
      self.bn=nn.BatchNorm1d(input_dim)
      self.input_to_h=nn.Linear(input_dim, hidden_dim)
      
      self.hidden_layers=nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ELU()) for i in range(n_layers)])
      self.z=nn.Linear(hidden_dim,z_dim)
      

   
    def forward(self,z_prev,a_t,h_t,constants):
  
      # concatenate the latent z and actions along the frequency dimension
      input_=torch.cat([z_prev,a_t,h_t,constants],dim=1)
        #B*K*(D+T) let's call D+T=F
      
      # input_=self.bn(input_)
      input_=self.input_to_h(input_)
      
      for layer in self.hidden_layers:
        input_=layer(input_)     #B*K*H
      

      
      z=self.z(input_)
                           
      return z

class state_space(nn.Module):
  
  def __init__(self,z_dim,obs_dim,full_dim,h_dim,a_dim,constants_dim):
    
    super(state_space,self).__init__()

    self.rnn=Encoder(full_dim,h_dim,a_dim)
    self.trans=Transition(z_dim,a_dim,h_dim,constants_dim)
    self.z_baselines=torch.FloatTensor([70.0,1.0,2.0,0.7,70.0]).to(device)
    self.initial=patient_encoder(constants_dim,z_dim,h_dim)
    
    if device=='cuda':
      self.cuda()

  def forward(self,trajectories,treatments,constants,mask,seq_lens,obs):
      """
      trajectories : Batch of Full_observations B*T*|Full_O|
      seq_lengths :list
      actions : B*T*|A|
      Constants B*|C|
      obs : B*T*|O|
      others : B*T*(|Full_O|-|O|)
      """
      
      T_max =trajectories.size(1)
      batch_size=trajectories.size(0)
      baselines=self.z_baselines.expand(batch_size,-1)
      z_prev,h_prev=self.initial(constants)
    
      h_prev=h_prev.contiguous()
      
      outputs=[]

      a_prev=torch.zeros(treatments.shape[0],treatments.shape[2]).to(device) #B*A
      rnn_output=self.rnn(trajectories,seq_lens,constants,h_prev) 
   
       #rnn_ouput has shape B*T*H

      # we enclose all the sample statements in the model in a plate.
      # this marks that each datapoint is conditionally independent of the others
      
      for t in range(1,T_max+1):

        z_t=self.trans(z_prev,a_prev,rnn_output[:,t-1,:],constants)
        z_scaled=z_t+baselines
        
        o_t=mechanistic(z_scaled) #B*obs_dim
        outputs.append(o_t)
        
        z_prev = z_t
        a_prev=treatments[:,t-1,:]
         
           
      temp=torch.stack(outputs,dim=1).to(device)
      return temp  #Batch_size*T*Obs_dim
