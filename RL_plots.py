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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda' if torch.cuda.is_available() else 'cpu'


import plotly.express as px
import plotly.graph_objects as go
def plot_polar_plots(pat,save=False):

    pat_df=df[df.Pat==pat]
    state=torch.Tensor(pat_df.iloc[:,:41].values)
    exps=model.get_exp_vals(state).squeeze(-1).cpu().detach().numpy()

    actions = ['Vaso :' +str(i)+' Fluids :'+ str(j) for i in range(3) for j in range(3)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
      r=(exps[0,:]+18)/36,
      theta=actions,
      # fill='toself',
      name='Admission'
      
    ))
    
    fig.add_trace(go.Scatterpolar(
      r=(exps[-12,:]+18)/36,
      theta=actions,
      # fill='toself',
      name='12 hours from death/Discharge'
      
    ))

    fig.add_trace(go.Scatterpolar(
      r=(exps[-5,:]+18)/36,
      theta=actions,
      # fill='toself',
      name='5 hours from actual Death/Discharge'
    
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
def plot_feature_uncertainty(pat,df,feat=None,size=(20,20)):
             
    

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
     
  cols=['coral','green','darkgrey']
        
        
  
  if feat:
    for i,feature in enumerate(feat):
      fig.add_trace(go.Scatter(x=np.arange(pat_df.shape[0]),y=pat_df[feature].values,
                    mode='lines',
                    name=feature,marker=dict(color=cols[i])))


  fig.show()
  
  
  

def plot_feature_ensm_treat(pat,df,feature='SBP',size=(10,10),lam=False):
  
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
  
  
  import matplotlib.pyplot as plt

def plot_feature_ensemble_exp_values(pat,feat=None,size=(20,20)):
  
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
  
  
  
  
 
  
  
