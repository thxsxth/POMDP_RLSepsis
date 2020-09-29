import torch
import numpy as np
import pandas as pd
import datetime as dt
import random
import time

device='cuda' if torch.cuda.is_available() else 'cpu'

df=pd.read_csv('RL_2.csv')
df=df.iloc[:,1:]

temp=df.describe()
means=temp.loc['mean'].values
stds=temp.loc['std'].values

cols=[0,1,2,3,4,5,6,7,11,38,40,41,42]

df.iloc[:,cols]=(df.iloc[:,cols]-means[cols])/stds[cols]

class DRL_dataset(Dataset):
  """
   Need the df Ready and scaled
   Normalization is Done before

  """

  def __init__(self,scale_rewards=False,df=df):
    self.scale_rewards=scale_rewards
    self.df=df

  def __len__(self):
    return df.shape[0]

  def __getitem__(self,idx):
    temp=self.df.iloc[idx,:].values
    # Rewards is always +- 15 at the terminal step and 
    done=int(np.abs(temp[-1])==15)
    states=torch.FloatTensor(temp[:41]).to(device)
    assert states.shape==(41,)
    if done:
      next_states=torch.zeros_like(states).to(device)
    else:
      next_states=torch.FloatTensor(self.df.iloc[idx+1,:].values[:41]).to(device)

    assert next_states.shape==(41,)

    reward=temp[-1]
    if self.scale_rewards:
      reward=(reward-2.313230e-02)/9.304778e-01
    
    actions=torch.FloatTensor(temp[-3:-1]).to(device)

    assert actions.shape==(2,)

    return states,next_states,actions,reward,done
    
