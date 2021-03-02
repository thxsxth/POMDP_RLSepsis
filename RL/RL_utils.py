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
device='cuda' if torch.cuda.is_available() else 'cpu'

def get_losses(Q,logQ,logQ1):
  """
  Q should be the reference

  """
  kld=torch.sum(torch.mul(Q, logQ-logQ1),-1)
  cross=-torch.sum(torch.mul(Q, logQ1),-1)
  
  return cross,kld

def get_losses_from_loader(loader,model_base,model_2,k=0):
  """
  model_2 and model_base should both output distributions
  loader should have a batch (of tuples) and batch[0]==state
  """
  ce=[]
  kl=[]
  for i,(batch) in enumerate(loader):
    with torch.no_grad():
      state=batch[0]
      Q,logQ=model_base.Q(state)
      Q1,logQ1=model_2.Q(state)
      cross,kld=get_losses(Q,logQ,logQ1)
    
    ce.append(cross.cpu().numpy())
    kl.append(kld.cpu().numpy())
    print('EPOCH :',k,i/(len(loader)),' DONe')
  return ce, kl

def get_actions(df):
  df['Fluids']=df['Fluids'].apply(lambda x:max(x,0))
  Vaso_cuts,vaso_bins=pd.cut(df['Vaso'],bins=[-1e-6,1e-8/2,0.15,np.inf],labels=False,retbins=True)

  df['Vaso_cuts']=Vaso_cuts.values
  Fluids_cuts,fluid_bins=pd.cut(df['Fluids'],bins=[-1e-6,6.000000e-03/2,5.000000e+01,np.inf],labels=False,retbins=True)

  df['Fluid_cuts']=Fluids_cuts.values
  df['Fluid_cuts'].value_counts(normalize=True),df['Vaso_cuts'].value_counts(normalize=True)

  all_acts=np.arange(9).reshape(3,3)


  df['action']=all_acts[df.Vaso_cuts,df.Fluid_cuts]



class Discrete_RL_dataset(Dataset):
  """
   Need the df Ready and scaled
   Normalization is Done before

  """

  def __init__(self,df,k,scale_rewards=False):
    self.scale_rewards=scale_rewards
    self.df=df
    self.k=k

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self,idx):
    
    temp=self.df.iloc[idx,:]
    # Rewards is always +- 15 at the terminal step and 
    done=int(np.abs(temp['Rewards'])==15)
    states=torch.FloatTensor(temp.iloc[:self.k].values).to(device)
    
    assert states.shape==(self.k,)
    if done:
      next_states=torch.zeros_like(states).to(device)
    
    else:
      next_states=torch.FloatTensor(self.df.iloc[idx+1,:].values[:self.k]).to(device)

    assert next_states.shape==(self.k,)

    reward=temp['Rewards']
    
    if self.scale_rewards:
      reward=(rewards-2.313230e-02)/9.304778e-01
    
    action=int(temp.action)

    # assert actions.shape==(2,)

    return states,next_states,action,reward,done


class corr_Dataset(Dataset):
  def __init__(self,df,feats=['SOFA_'],k=41):
    self.df=df
    self.feats=feats
    self.k=k

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self,idx):
    return torch.Tensor(self.df.iloc[idx,:][: self.k].values).to(device), torch.FloatTensor(self.df.iloc[idx,:][self.feats].values).to(device)