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


class Discrete_RL_dataset(Dataset):
  """
   Need the df Ready and scaled
   Normalization is Done before

  """

  def __init__(self,scale_rewards=False,df=df,k=41):
    self.scale_rewards=scale_rewards
    self.df=df
    self.k=k

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self,idx):
    
    temp=self.df.iloc[idx,:]
    # Rewards is always +- 15 at the terminal step and 
    done=int(np.abs(temp['Rewards'])==15 )
    
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

def get_times_to_deaths(model,df,k=41):
  """
  Returns times to deaths of pattients, and Qs (Returns model.Q.(state))

  """
  time_to_deaths=[]
  Qs=[]
  for i,pat in enumerate(df.Pat.unique()):
    pat_df=df[df.Pat==pat]
    dead=pat_df.Rewards.iloc[-1]==-15
    if dead:
      death_times=np.arange(pat_df.shape[0],0,-1)
    else :
      death_times=np.zeros(pat_df.shape[0])
    time_to_deaths.append(death_times)
    states=torch.FloatTensor(pat_df.iloc[:,:k].values).to(device)
  
    with torch.no_grad():
      Q,_=model.Q(states)
      
  
    Qs.append(Q[:,0,:].cpu().numpy())
  
    print(i/df.Pat.unique().shape[0])

  return time_to_deaths, Qs

def eval_policy_wis(policy, behav_cloner, df,df_b,clinician=False,ensemble=False,gam=None,
                    
                    random=False,dead_only=False,per=False,p=1):


	
  wis_returns = 0
  wis_weighting = 0
  state_dim=policy.state_dim
  state_dim_b=behav_cloner.state_dim


  for it,pat in enumerate(df.Pat.unique()):
    pat_df=df[df.Pat==pat]
    #For behavior let's not consider latent states

    pat_df_b=df_b[df_b.Pat==pat]
    
    states=torch.FloatTensor(pat_df.iloc[:,:state_dim].values).to(device)
    states_b=torch.FloatTensor(pat_df_b.iloc[:,:state_dim_b].values).to(device)
    
    actions=torch.LongTensor(pat_df.action.values).to(device)
    if dead_only:
      rewards=torch.zeros_like(actions)
      rewards[-1]=pat_df.Rewards.values[-1]
    else:
       rewards=torch.FloatTensor(pat_df.Rewards.values)
    
    if gam:
     gamma=torch.FloatTensor([gam**t for t in range(pat_df.shape[0])])
     rewards=rewards*gamma
	

    with torch.no_grad():
      _,i=behav_cloner(states_b)
      p_b=F.softmax(i,dim=1)
      p_b=torch.gather(p_b,1,actions.unsqueeze(-1)).squeeze(-1)
      
      if ensemble:
        Q=policy.get_ensemble_exp_values(states)
      elif per:
        Q=policy.get_percentile(states,p)
      
      else :
        Q=policy.get_exp_vals(states) #Q.shape T*9*1
        # Q=Q*(Q/Q.max(1, keepdim=True)[0]>=1).float()+Q*0.01
        # Q=Q*(Q/Q.max(1, keepdim=True)[0]>=1).float()

       
        
      p_pol=F.softmax(Q,dim=1).squeeze(-1)
      p_pol=torch.gather(p_pol,1,actions.unsqueeze(-1)).squeeze(-1)
      
      if random:
        p_pol=torch.ones_like(actions)*1/9.0

       

    
			
		
    if not (p_b > 0).all(): 
	    p_b[p_b==0] = 0.1

		
	
    if clinician:
      cum_ir =torch.tensor(1.0)
    else:
      cum_ir = torch.clamp((p_pol / p_b).prod(), 1e-30, 1e4)
    # cum_ir = torch.clamp((torch.ones_like(p_b)).prod(), 1e-30, 1e4)

  
    wis_rewards = cum_ir.cpu()*rewards.sum().item()
    wis_returns+=wis_rewards
    wis_weighting+=cum_ir
	  


    # print(it/df.Pat.unique().shape[0])
  wis_eval = (wis_returns / wis_weighting) 
  return wis_eval

def eval_policy_2(policy, behav_cloner, df,df_b,clinician=False,ensemble=False,gam=None,per=False,p=1):


	
  wis_returns = 0
  wis_weighting = 0
  state_dim=policy.state_dim
  state_dim_b=behav_cloner.state_dim


  for it,pat in enumerate(df.Pat.unique()):
    
    pat_df=df[df.Pat==pat]
    pat_df_b=df_b[df_b.Pat==pat]
    
    states=torch.FloatTensor(pat_df.iloc[:,:state_dim].values).to(device)
    states_b=torch.FloatTensor(pat_df_b.iloc[:,:state_dim_b].values).to(device)
    
    actions=torch.LongTensor(pat_df.action.values).to(device)
    rewards=torch.FloatTensor(pat_df.Rewards.values)

    # rewards=torch.zeros_like(actions)
    # rewards[-1]=pat_df.Rewards.values[-1]
    
    rewards=torch.FloatTensor(pat_df.Rewards.values).to(device)

    if gam:
    
      gamma=torch.FloatTensor([gam**t for t in range(pat_df.shape[0])])
      rewards=rewards*gamma
		# Evaluate the probabilities of the observed action according to the trained policy and the behavior policy

    with torch.no_grad():
      _,i=behav_cloner(states_b)
      p_b=F.softmax(i,dim=1)
      p_b=torch.gather(p_b,1,actions.unsqueeze(-1)).squeeze(-1)
      

      if ensemble:
        Q=policy.get_ensemble_exp_values(states)

      elif per:
        Q=policy.get_percentile(states,p)
      
      else :
        Q=policy.get_exp_vals(states) #Q.shape T*9*1
        
      # p_pol=F.softmax(Q,dim=1).squeeze(-1)
      # p_pol=0.99*mask*p_pol+(1-mask)*(0.1/8)*p_pol
      mask=(Q/Q.max(1, keepdim=True)[0]>=1).float().squeeze(-1)   
      p_pol=0.99*mask+(0.01/8)*(1-mask)
      p_pol=torch.gather(p_pol,1,actions.unsqueeze(-1)).squeeze(-1)


       

    
		
    if not (p_b > 0).all(): 
	    p_b[p_b==0] = 0.1

		
	
    if clinician:
      cum_ir =torch.tensor(1.0)
    else:
      cum_ir = torch.clamp((p_pol / p_b).prod(), 1e-30, 1e4)
      # cum_ir = torch.clamp((p_pol).prod(), 1e-30, 1e4)
    # cum_ir = torch.clamp((torch.ones_like(p_b)).prod(), 1e-30, 1e4)

  
    wis_rewards = cum_ir.cpu()*rewards.sum().item()
    wis_returns+=wis_rewards
    wis_weighting+=cum_ir
	  


    # print(it/df.Pat.unique().shape[0])
  wis_eval = (wis_returns / wis_weighting) 
  return wis_eval