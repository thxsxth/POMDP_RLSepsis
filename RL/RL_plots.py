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

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from distRL import dist_DQN,ensemble_distDQN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda' if torch.cuda.is_available() else 'cpu'
print('Done')


df=pd.read_csv('RL_new.csv')
df=df.iloc[:,1:]
print('Loaded')
df['SOFA_']=df.SOFA
df['SV_']=df.SV
df['SV_C']=(df.SV_/df.C)
df['CO']=(df.SV*df.HR)
df['CO_C']=df.CO/df.C

df['SV_C']=(df['SV_C']-df['SV_C'].mean())/(df['SV_C'].std())
df['CO']=(df['CO']-df['CO'].mean())/(df['CO'].std())
df['CO_C']=(df['CO_C']-df['CO_C'].mean())/(df['CO_C'].std())


df['SV/C']=df['SV_C']
df['CO/C']=df['CO_C']
df['RC']=df['R']*df['C']
df['SV_R']=df['R']*df['SV_']
df['SV_R']=(df['SV_R']-df['SV_R'].mean())/df['SV_R'].std()


temp=df.describe()
means=temp.loc['mean'].values
stds=temp.loc['std'].values

cols=[0,1,2,3,4,5,6,7,11,37,38,40]

df.iloc[:,cols]=(df.iloc[:,cols]-means[cols])/stds[cols]
print('Normalized')


def get_actions(df):
  df['Fluids']=df['Fluids'].apply(lambda x:max(x,0))
  Vaso_cuts,vaso_bins=pd.cut(df['Vaso'],bins=[-1e-6,1e-8/2,0.15,np.inf],labels=False,retbins=True)

  df['Vaso_cuts']=Vaso_cuts.values
  Fluids_cuts,fluid_bins=pd.cut(df['Fluids'],bins=[-1e-6,6.000000e-03/2,5.000000e+01,np.inf],labels=False,retbins=True)

  df['Fluid_cuts']=Fluids_cuts.values
  df['Fluid_cuts'].value_counts(normalize=True),df['Vaso_cuts'].value_counts(normalize=True)

  all_acts=np.arange(9).reshape(3,3)


  df['action']=all_acts[df.Vaso_cuts,df.Fluid_cuts]


get_actions(df)

ensm=ensemble_distDQN()

import matplotlib.pyplot as plt
def plot_feature_ensemble_exp_values(pat,df=df,feat=None,size=(20,20)):
  
  legends=['Vaso :' +str(i)+' Fluids :'+ str(j) for i in range(3) for j in range(3)]
  colors=['cornflowerblue','royalblue','midnightblue','lightcoral','indianred','darkred','orange','chocolate','grey']  
             
    
  plt.figure(figsize=size)
  pat_df=df[df.Pat==pat]

  state=torch.FloatTensor(df[df.Pat==pat].iloc[:,:41].values).to(device)
  
  exps=ensm.get_ensemble_exp_values(state) #T*9

  for k in range(9):
     plt.plot(np.arange(state.shape[0]),exps[:,k].cpu().detach().numpy(),color=colors[k],linestyle='dotted',linewidth=1,label=legends[k],marker=".",markersize=4)

  if feat:
     plt.plot(np.arange(state.shape[0]),pat_df[feat].values,'g',linestyle=':',linewidth=1,marker='*',label='Standardized {}'.format(feat))
     plt.hlines(df[feat].mean(),0,(pat_df.shape[0]),'k',label='Mean of {}'.format(feat))

  
  
  plt.legend()
  plt.show()
  

def plot_feature_ensm_treat(pat,df=df,feature='SBP',size=(10,10),lam=False):
  
  legends = ['Fluids for the Patient', 'Standardized {}'.format(feature),'Mean {}'.format(feature),'Vaso for the patient']
  acts=[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]          
  
  plt.figure(figsize=size)
  pat_df=df[df.Pat==pat]

  state=torch.FloatTensor(df[df.Pat==pat].iloc[:,:41].values).to(device)

  action=ensm.get_ensemble_action(state,lam=lam)
  fluids_pat=[acts[action[i]][1] for i in range(action.shape[0])]
  vaso_pat=[acts[action[i]][0] for i in range(action.shape[0])]

  plt.plot(np.arange(pat_df.shape[0]),fluids_pat,'royalblue',linestyle='dotted',linewidth=0.8,label=legends[0],marker=".",markersize=5)
  plt.plot(np.arange(pat_df.shape[0]),vaso_pat,'orange',linestyle='dotted',linewidth=0.8,label=legends[3],marker=".",markersize=5)
  
  plt.plot(np.arange(pat_df.shape[0]),pat_df[feature].values,'g',linestyle=':',linewidth=1,marker='*',label=legends[1])
  title='Lambda :'+ str(lam)
  plt.title(title)
  
   
  plt.hlines(df[feature].mean(),0,(pat_df.shape[0]),'k',label=legends[2])
  
  plt.legend()
  
import plotly.graph_objects as go

# actions = ['Vaso :' +str(i)+' Fluids :'+ str(j) for i in range(3) for j in range(3)]
# def plot_feature_uncertainty(pat,df=df,feat=None,size=(20,20)):
             
    

#   pat_df=df[df.Pat==pat]
#   state=torch.FloatTensor(df[df.Pat==pat].iloc[:,:41].values).to(device)
  
#   exps=model.get_exp_vals(state).squeeze(-1).detach().cpu().numpy()  #T*9
#   cols=['Uncertainty_act_{}'.format(i) for i in range(9)]
#   uncertains=pat_df[cols].values
  
#   fig = go.Figure()
#   for k in range(9):
#      label='Expected Values for '+ actions[k]
    
#     #  label="Expected Values for Action : "+actions[k]
#      fig.add_trace(go.Scatter(x=np.arange(pat_df.shape[0]), y=exps[:,k],
#                     mode='lines+markers', marker=dict(size=uncertains[:,k]*10),
#                     name=label))
     
#   cols=['coral','green','darkgrey']
        
        
  
#   if feat:
#     for i,feature in enumerate(feat):
#       fig.add_trace(go.Scatter(x=np.arange(pat_df.shape[0]),y=pat_df[feature].values,
#                     mode='lines',
#                     name=feature,marker=dict(color=cols[i])))


#   fig.show()
 
  
  
kl=pd.read_csv('avergae_KLs.csv')
for i in range(9):
  
  df['kl_uncertainty_for_act_{}'.format(i)]=kl.iloc[:,i]
  
  

def plot_polar_plots(pat,save=False,df=df,times=[0,-12,-5]):

    pat_df=df[df.Pat==pat]
    state=torch.Tensor(pat_df.iloc[:,:41].values)
    exps=model.get_exp_vals(state).squeeze(-1).cpu().detach().numpy()

    actions = ['Vaso :' +str(i)+' Fluids :'+ str(j) for i in range(3) for j in range(3)]


    for time in times:
      fig = go.Figure()
      name='Admission' if time==0 else '{} hours from death'.format(-time)
      if time>0:
        name='{} hours from admission'.format(time)
      fig.add_trace(go.Scatterpolar(
        r=(exps[time,:]+18)/36,
        theta=actions,
        # fill='toself',
        name=name
      
          ))
    
   
    fig.update_layout(
      polar=dict(
      radialaxis=dict(
      visible=True,
      range=[0, 1],
      showline=True
      
      )),
      showlegend=True

    )

    fig.show()
    if save==True:
      # fig.write_html("'radarplot_{}.html".format(pat))
      fig.write_image("radarplot_{}.jpg".format(pat))
      
      


actions = ['Vaso :' +str(i)+' Fluids :'+ str(j) for i in range(3) for j in range(3)]
def plot_feature_uncertainty(pat,df=df,feat=None,size=(20,20),marker_only=False):
             
    

  pat_df=df[df.Pat==pat]
  state=torch.FloatTensor(df[df.Pat==pat].iloc[:,:41].values).to(device)
  
  exps=model.get_exp_vals(state).squeeze(-1).detach().cpu().numpy()  #T*9
  cols=['Uncertainty_act_{}'.format(i) for i in range(9)]
  uncertains=pat_df[cols].values
  
  fig = go.Figure()
  for k in range(9):
     label='Expected Values for '+ actions[k]
    
    #  label="Expected Values for Action : "+actions[k]
     fig.add_trace(go.Scatter(x=np.arange(pat_df.shape[0]), y=exps[:,k],
                    mode='lines+markers', marker=dict(size=uncertains[:,k]*10),
                    name=label))
     
  cols=['peru','darkslategray','darkcyan']
        
        
  
  if feat:
    for i,feature in enumerate(feat):
      style='lines+markers'
      if marker_only:
        style='markers'
      if feature=='SOFA':
        style='lines+markers'
      fig.add_trace(go.Scatter(x=np.arange(pat_df.shape[0]),y=pat_df[feature].values,
                    mode=style,
                    name=feature,marker=dict(color=cols[i],size=4,symbol='x')))


  fig.show()
 
  
  

      
      
 
 


