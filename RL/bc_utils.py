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
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda' if torch.cuda.is_available() else 'cpu'

def get_actions(df):
  df['42']=df['42'].apply(lambda x:max(x,0))
  Vaso_cuts,vaso_bins=pd.cut(df['41'],bins=[-1e-6,0.000900/2,0.067500,0.142469,0.408000,np.inf],labels=False,retbins=True)

  df['Vaso_cuts']=Vaso_cuts.values
  Fluids_cuts,fluid_bins=pd.cut(df['42'],bins=[-1e-6,6.000000e-03/2,1.000000e+01,5.000000e+01,2.107170e+02,np.inf],labels=False,retbins=True)

  df['Fluid_cuts']=Fluids_cuts.values
  df['Fluid_cuts'].value_counts(normalize=True),df['Vaso_cuts'].value_counts(normalize=True)

  all_acts=np.arange(25).reshape(5,5)


  df['action']=all_acts[df.Vaso_cuts,df.Fluid_cuts]


def train_epoch(model,data_loader,optimizer,it=0):
    model.train()
    sum_i_loss=0
    for batch,(state,next_state,action,reward,done) in enumerate(data_loader):
        batch_size=state.shape[0]
        action=torch.LongTensor(action.reshape(batch_size,1)).to(device)
        # reward=torch.FloatTensor(reward.reshape(batch_size,1).cpu().numpy()).to(device)
        # done=done.reshape(batch_size,1).to(device)
          
          
        (imt, i)= model(next_state)

        # imt = imt.exp()
        i_loss = F.nll_loss(imt, action.reshape(-1))
        
        # print(i_loss.item())

        sum_i_loss+=i_loss.item()

        if batch%25==0:
          print('Epoch :',it,'Batch :', batch,'Average i Loss :',sum_i_loss/(batch+1))

        optimizer.zero_grad()
        i_loss.backward()
        optimizer.step()

        
def evaluate(model,data_loader):
    model.eval()
    sum_i_loss=0
    for batch,(state,next_state,action,reward,done) in enumerate(data_loader):
      with torch.no_grad():
        batch_size=state.shape[0]
        action=torch.LongTensor(action.reshape(batch_size,1)).to(device)
        # reward=torch.FloatTensor(reward.reshape(batch_size,1).cpu().numpy()).to(device)
        # done=done.reshape(batch_size,1).to(device)
                    
        (imt, i)=model(next_state)
        # imt = imt.exp()
        i_loss = F.nll_loss(imt, action.reshape(-1))

        sum_i_loss+=i_loss.item()

        if batch%25==0:
          print('Epoch :',it,'Batch :', batch,'Average i Loss :',sum_i_loss/(batch+1))


def Normalize(df,scaler):
    df1=scaler.transform(df.iloc[:,:-4])
    df1['43']=df['43']
    df1['action']=df['action']

    return df1





         