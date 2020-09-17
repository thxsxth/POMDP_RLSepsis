import torch
import numpy as np 
import pandas as pd  

import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels


def plot_cv_state(seq_len,vaso,fluids,cvs):

  """

  seq_len : The seq lens of the patient
  Vaso: Vasopressors Sequence : torch.tensor
  Fluids: Fluids Sequence torch.tensor
  cvs : CV (Hidden Varaibles) outputs from the model torch.tensor

  Plots SV/C and RC, SVR

  """
  plt.figure(figsize=(30,30))
  plt.subplot(3, 2, 1)
  plt.title('Vasopressor Dosage', fontsize=20)

  plt.plot(np.arange(seq_len),vaso.numpy(),marker='o',color='g',linestyle='dotted',linewidth=2)

  plt.subplot(3, 2, 2)
  plt.title('Fluids', fontsize=20)
  plt.plot(np.arange(seq_len),fluids.numpy(),marker='.',color='b',linestyle='dotted',linewidth=2)

  plt.subplot(3, 2, 3)
  plt.title('SV/C', fontsize=20)
  plt.plot(np.arange(seq_len),(cvs[:,4]/cvs[:,2]).numpy(),marker='x',color='k',linestyle='dotted',linewidth=2)


  plt.subplot(3, 2, 4)
  plt.title('RC', fontsize=20)
  plt.plot(np.arange(seq_len),(cvs[:,2]*cvs[:,1]).numpy(),marker='x',color='r',linestyle='dotted',linewidth=2)

  plt.subplot(3, 2, 5)
  plt.title('(CO)*R', fontsize=20)
  plt.plot(np.arange(seq_len),(cvs[:,0]*cvs[:,4]*cvs[:,1]).numpy(),marker='x',color='r',linestyle='dotted',linewidth=2)

  plt.subplot(3, 2, 6)
  plt.title('(SV)*R', fontsize=20)
  plt.plot(np.arange(seq_len),(cvs[:,4]*cvs[:,1]).numpy(),marker='x',color='r',linestyle='dotted',linewidth=2)

  plt.show()
  # plt.savefig('complete_cv_state.jpg')


def plot_cv_paramters(seq_len,vaso,fluids,cvs):

  """
  seq_len : The seq lens of the patient
  Vaso: Vasopressors Sequence : torch.tensor
  Fluids: Fluids Sequence torch.tensor
  cvs : CV (Hidden Varaibles) outputs from the model torch.tensor

  """
  plt.figure(figsize=(30,30))
  plt.subplot(3, 2, 1)
  plt.title('Vasopressor Dosage', fontsize=20)

  plt.plot(np.arange(seq_len),vaso.numpy(),marker='o',color='g',linestyle='dotted',linewidth=2)

  plt.subplot(3, 2, 2)
  plt.title('Fluids', fontsize=20)
  plt.plot(np.arange(seq_len),fluids.numpy(),marker='.',color='b',linestyle='dotted',linewidth=2)

  plt.subplot(3, 2, 3)
  plt.title('Resistance', fontsize=20)
  plt.plot(np.arange(seq_len),cvs[:,1].numpy(),marker='x',color='k',linestyle='dotted',linewidth=2)


  plt.subplot(3, 2, 4)
  plt.title('Compliance', fontsize=20)
  plt.plot(np.arange(seq_len),cvs[:,2].numpy(),marker='x',color='r',linestyle='dotted',linewidth=2)

  plt.subplot(3, 2, 5)
  plt.title('Heart Rate', fontsize=20)
  plt.plot(np.arange(seq_len),cvs[:,0].numpy(),marker='x',color='r',linestyle='dotted',linewidth=2)

  plt.subplot(3, 2, 6)
  plt.title('SV', fontsize=20)
  plt.plot(np.arange(seq_len),cvs[:,4].numpy(),marker='x',color='r',linestyle='dotted',linewidth=2)

  plt.show()



def plot_outputs(seq_len,obs,out):
  titles=['Heart Rate','Systolic BP','Diastolic BP','Mean BP']
  fig = plt.figure(figsize=(50,50))
  for i in range(1, 5):
    
    ax = fig.add_subplot(2, 2, i)
    # ax.scatter(obs1[:,i-1].numpy(),out1[:,i-1].detach().numpy(),marker='x')
    
    ax.plot(np.arange(seq_len),obs[:,i-1].numpy(),marker='x',color='r',linestyle='dotted',linewidth=3)
    ax.plot(np.arange(seq_len),out[:,i-1].detach().numpy(),marker='o',color='g',linestyle='dotted',linewidth=3)
    plt.title(titles[i-1], fontsize=30)
    # plt.tick_params(axis='x', colors='white')
    # plt.tick_params(axis='y', colors='white')
    


