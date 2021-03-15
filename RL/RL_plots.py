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
import seaborn as sns

import matplotlib.pyplot as plt
# from distRL import *

device='cuda' if torch.cuda.is_available() else 'cpu'

import matplotlib.pyplot as plt
def plot_feature_exp_values(pat,df,model,ensm,feat=None,size=(10,10),filename=None):
  """
  model should have model.get_exp_vals method
  ensm should have get_uncertainty method
  df : Pandas.DataFrame and first state_dim columns should be the state

  """

  
  legends=['Vaso :' +str(i)+' Fluids :'+ str(j) for i in range(3) for j in range(3)]
  colors=['cornflowerblue','royalblue','midnightblue','gold','orange','tomato','lightcoral','crimson','darkred']  
  feat_colors=['darkgreen','deeppink','darkturquoise']
    
  plt.figure(figsize=size)
  pat_df=df[df.Pat==pat]
  state_dim=model.state_dim
  state=torch.FloatTensor(df[df.Pat==pat].iloc[:,:state_dim].values).to(device)
  
  exps=model.get_exp_vals(state).squeeze(-1)  #T*9
  uncertains=ensm.get_uncertainty(state).detach().cpu().numpy()

  for k in range(9):
     plt.plot(np.arange(state.shape[0]),exps[:,k].cpu().detach().numpy(),color=colors[k],linestyle='-',linewidth=0.6,label=legends[k],marker=".")
     plt.scatter(np.arange(state.shape[0]),exps[:,k].cpu().detach().numpy(),color=colors[k],linewidths=1,marker=".",s=uncertains[:,k]*200)

  if feat:
    for j,feature in enumerate(feat):
     plt.plot(np.arange(state.shape[0]),pat_df[feature].values,linestyle=':',linewidth=2,marker='*',label='Standardized {}'.format(feature),markersize=5,color=feat_colors[j])
    #  plt.hlines(df[feature].mean(),0,(pat_df.shape[0]),label='Mean of {}'.format(feature))

  
  plt.xlabel('Time Step (From ICU admission in hours)',fontsize=15)
  plt.ylabel('Expected Value/Feature Value',fontsize=15)
  title='Evolution of expected values of value distributions ' +'ICU stay ID :' +str(pat)
  plt.title(title,fontsize=20)
  plt.legend()
  plt.show()
  if filename:
    plt.savefig(filename)


def plot_outputs(pat,params,df,model,ensm,save_file=False):
  """
  
  model should have model.get_exp_vals method
  ensm should have get_uncertainty method
  df : Pandas.DataFrame and first state_dim columns should be the state


  """
  fig = plt.figure(figsize=(60,20))
  pat_df=df[df.Pat==pat]
  state_dim=model.state_dim
  state=torch.FloatTensor(df[df.Pat==pat].iloc[:,:state_dim].values).to(device)
  acts=[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)] 
  plt.subplots_adjust(left=None, bottom=0.1, right=None, top=3, wspace=None, hspace=None)
  for i in range(1, len(params)+1):
    
    ax = fig.add_subplot(5,2, i)
    legends = ['Fluids for the Patient','Vaso for the patient']
    
    
    action,_=ensm.uncertainty_aware_actions(state,params[i-1][0],params[i-1][1],params[i-1][2])
    fluids_pat=[acts[action[i]][1] for i in range(action.shape[0])]
    vaso_pat=[acts[action[i]][0] for i in range(action.shape[0])]

    ax.plot(np.arange(pat_df.shape[0]),fluids_pat,'k',linestyle='dashed',linewidth=3.5,label=legends[0],marker=".",markersize=8)
    ax.plot(np.arange(pat_df.shape[0]),vaso_pat,'r',linestyle='dashed',linewidth=3.5,label=legends[1],marker=".",markersize=8)
    ax.set_yticks([0,1,2])
    ax.tick_params(labelsize=30)

    beta1,beta2,lam=params[i-1]
    title=r'$\beta_{1}$ :' +str(beta1)+' , '+ r'$\beta_{2}$ :' + str(beta2)+' , '+r'$\lambda$ :'+str(lam)
    plt.title(title,fontdict={'fontsize':45})
    plt.legend(fontsize=25)
    plt.xlabel('Time Step (from ICU admission)', fontsize=30)
    plt.ylabel('Action', fontsize=30)

  if save_file:
    plt.savefig(save_file)



def plot_clinician_actions(pat,df,size=(20,10),save_title=False):
  acts=[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]          
  
  plt.figure(figsize=size)
  pat_df=df[df.Pat==pat]
  clinician_action=pat_df.action.values
  
#   state=torch.FloatTensor(df[df.Pat==pat].iloc[:,:41].values).to(device)

  fluids_clinician=[acts[clinician_action[i]][1] for i in range(clinician_action.shape[0])]
  vaso_clinician=[acts[clinician_action[i]][0] for i in range(clinician_action.shape[0])]


  plt.plot(np.arange(pat_df.shape[0]),fluids_clinician,'green',linestyle='dashed',linewidth=1.5,label=' Clinician Fluids',marker=".",markersize=8)
  plt.plot(np.arange(pat_df.shape[0]),vaso_clinician,'blue',linestyle='dashed',linewidth=1.5,label=' Clinician Vaso',marker=".",markersize=8)
  plt.yticks([0,1,2])

  plt.legend(fontsize='x-large',loc=0)
  plt.tick_params(labelsize=20)
  plt.title('Vasopressors and fluids administrated by clinician',fontdict={'fontsize':25})

  if save_title:
    filename='{}.jpg'.format(save_title)
    plt.savefig(filename)



import plotly.express as px
import plotly.graph_objects as go
def plot_polar_plots(pat,df,model,save=False,times=[0,-12,-5]):

    pat_df=df[df.Pat==pat]
    state=torch.Tensor(pat_df.iloc[:,:41].values)
    exps=model.get_exp_vals(state).squeeze(-1).cpu().detach().numpy()

    actions = ['Vaso :' +str(i)+' Fluids :'+ str(j) for i in range(3) for j in range(3)]


  
    fig = go.Figure()
    names=['Admission']+['{} hours from death'.format(-time) for time in times[1:]]
    fig.add_trace(go.Scatterpolar(
        r=(exps[times[0],:]+18)/36,
        theta=actions,
        # fill='toself',
        name=names[0]
      
          ))
    
    fig.add_trace(go.Scatterpolar(
        r=(exps[times[1],:]+18)/36,
        theta=actions,
        # fill='toself',
        name=names[1]
      
          ))
    
    fig.add_trace(go.Scatterpolar(
        r=(exps[times[0],:]+18)/36,
        theta=actions,
        # fill='toself',
        name=names[2]
      
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




