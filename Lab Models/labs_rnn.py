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


class Encoder_labs(nn.Module):
  """ Encodes x_{:t} 
      The job of the Encoder is to remember the past observations.
     
  """

  def __init__(self,input_dim,h_dim,n_layers=12, dropout=0.0):
      
      super(Encoder_labs,self).__init__()
      
      self.rnn=nn.GRU(input_dim,h_dim,n_layers,batch_first=True)
      self.dropout=dropout  
      self.n_layers=n_layers
      self.hidden_dim=h_dim
      self.init_weights()
      

  def  init_weights(self):
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)

  def forward(self,obs,obs_lens,init_h=None, noise=False):

    
    batch_size, max_len, freq=obs.size()  
    obs_lens=torch.LongTensor(obs_lens).to(device)
    obs_lens_sorted, indices = obs_lens.sort(descending=True)
    obs_sorted = obs.index_select(0, indices).to(device)
    
    packed_obs=pack_padded_sequence(obs_sorted,obs_lens_sorted.data.tolist(),batch_first=True)
    # if init_h is None:
    #     init_h=self.init_h

     
    
    hids, h_n = self.rnn(packed_obs) # hids: [B x T x H]  
                                                  # h_n: [num_layers*B*H)
    _, inv_indices = indices.sort()

    hids, lens = pad_packed_sequence(hids, batch_first=True)         
    hids = hids.index_select(0, inv_indices) #B*T*H
    
    
            
    return hids

class stacked_autoencoder_new(nn.Module):
  """
  Denoising autoencoder adjusted so it will return the hidden state

  """

  def __init__(self,input_dim,h_dims=[512,128,10],n_layers=[12,8,3]):
    
    super(stacked_autoencoder_new,self).__init__()   
    self.encoder1=Encoder_labs(input_dim,h_dims[0],n_layers[0])
    self.encoder2=Encoder_labs(h_dims[0],h_dims[1],n_layers[1])
    self.encoder3=Encoder_labs(h_dims[1],h_dims[2],n_layers[2])
    self.linear=nn.Linear(h_dims[2],input_dim)


  def forward(self,obs,obs_lens):

    out1=self.encoder1(obs,obs_lens)  #B*T*h_1
    out1=pad_sequence(out1,batch_first=True)

    out2=self.encoder2(out1,obs_lens)  #B*T*h_1
    out2=pad_sequence(out2,batch_first=True) #B*T*h_2

    out3=self.encoder3(out2,obs_lens)
    out3=pad_sequence(out3,batch_first=True) #B*T*h_3
    
    output_=self.linear(out3)

    return output_,out3


