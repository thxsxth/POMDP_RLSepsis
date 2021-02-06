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

vitals=pd.read_csv('Vitals/Vitals.csv',parse_dates=['charttime']) #pivoted vitals
# sofa=pd.read_csv('../pivoted_sofa/pivoted_sofa.csv',parse_dates=['endtime','starttime']) #pivoted sofa
labs=pd.read_csv('pivoted_labs/Pivoted_labs.csv',parse_dates=['charttime'])
sofa=pd.read_csv('./pivoted_sofa/sofa_with_vaso3.csv',parse_dates=['endtime','starttime'])

#sofa['Vaso']=sofa['rate_norepinephrine'].add(0.1*sofa['rate_dopamine'],fill_value=0).add(sofa['rate_epinephrine'],fill_value=0)
#sofa['Vaso']=sofa['Vaso'].add(0.1*sofa['rate_dobutamine'],fill_value=0).add(10*sofa['rate_vasopressin'],fill_value=0).fillna(0)

vitals['TempC']=vitals['TempC'].ffill()
sofa['GCS_min']=sofa['GCS_min'].ffill()
labs['icustay_id']=labs['ICUSTAY_ID']

co=pd.read_csv('sepsis3_adults.csv',
               parse_dates=['intime','outtime','suspected_infection_time_poe']) #cohort + demographics
co=co.set_index('icustay_id')

admissions=pd.read_csv('admissions.csv',parse_dates=['ADMITTIME','DISCHTIME','DEATHTIME'])
admissions=admissions.set_index('icustay_id').sort_index()
co['death_time']=admissions['DEATHTIME']

vaso_cv=pd.read_csv('./pivoted_sofa/Vaso_CV.csv',parse_dates=['charttime'])
vaso_MV=pd.read_csv('./pivoted_sofa/Vaso_MV.csv',parse_dates=['endtime','starttime'])

vaso_cv['starttime']=vaso_cv['charttime']

fluids_cv=pd.read_csv('./pivoted_sofa/fluids_CV.csv',parse_dates=['charttime'])
fluids_MV=pd.read_csv('./pivoted_sofa/fluids_MV.csv',parse_dates=['endtime','starttime'])

fluids_cv['starttime']=fluids_cv['charttime']


# input_cv=pd.read_csv('./Fluids/cleaned_input_cv.csv',parse_dates=['charttime']) 
# input_mv=pd.read_csv('./Fluids/input_eventsMV.csv',parse_dates=['starttime','endtime'])

# input_cv=input_cv[['icustay_id','charttime','tev']]
# input_mv=input_mv[['icustay_id','endtime','tev']]
# input_mv['tev_mv']=input_mv['tev']
# input_mv['charttime']=input_mv['endtime']
# input_mv=input_mv.drop('tev',axis=1)
# input_fluids=input_mv.merge(input_cv,on=['icustay_id','charttime'],how='outer')[['icustay_id','charttime','tev','tev_mv']]


# input_fluids['volume']=input_fluids['tev'].add(input_fluids['tev_mv'],fill_value=0)


def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = torch.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0:seq_lengths[b]] = torch.ones(seq_lengths[b])
    return mask.to(device)

class modeling_dataset(Dataset):
  """
  Implements a dataset for patients
  Needs Vitals,Sofa,Inputs,co tables

  This is only used for Model Learning, Rewards, Terminals are unnecessary
  """
  def __init__(self,patient_ids,train=True,T_max=48):
    #patient_ids :List/np.array
    self.ids=patient_ids
    self.train=train
    self.T_max=T_max

  def __len__(self):
    return len(self.ids)

  def __getitem__(self,idx):
    #Get Patient from the index
    pat=self.ids[idx]
    # pat_fluids=input_fluids[input_fluids.icustay_id==pat].set_index('charttime')
    pat_sofa=sofa[sofa.icustay_id==pat].set_index('endtime').resample('H').last().ffill()
#     pat_sofa=pd.concat([pat_sofa,pat_fluids]).resample('H').sum()
    
    pat_vaso=pd.concat([vaso_cv[vaso_cv.icustay_id==pat].set_index('starttime').resample('H').max(),
           vaso_MV[vaso_MV.icustay_id==pat].set_index('starttime').resample('H').max().fillna(0)]).resample('H').max().fillna(0)
    
    #The two csv files had rate_std and vaso_rate to mean the same thing
    pat_vaso['vaso_rate']=np.max(pat_vaso['rate_std'],pat_vaso['vaso_rate'])

    pat_vaso=pat_vaso[['icustay_id','vaso_rate']]
    
    pat_fluids=pd.concat([fluids_cv[fluids_cv.icustay_id==pat].set_index('starttime').resample('H').sum(),
           fluids_MV[fluids_MV.icustay_id==pat].set_index('starttime').resample('H').sum().fillna(0)]).resample('H').last        ().fillna(0)

    pat_fluids['volume']=pat_fluids['tev_mv']+pat_fluids['tev_cv']

    pat_vitals=vitals[vitals.icustay_id==pat].set_index('charttime')
    pat_labs=labs[labs.icustay_id==pat]
    pat_df=pd.concat([pat_vitals,
                              pat_sofa,pat_vaso,pat_fluids]).resample('H').last()[['HeartRate','SysBP','DiasBP',	'MeanBP','RespRate','SpO2','TempC',
                                      'liver_24hours','cardiovascular_24hours',
                                      'cns_24hours','renal_24hours','SOFA_24hours','volume','vaso_rate']].resample('H').last()

    dead=co.loc[pat].HOSPITAL_EXPIRE_FLAG==1
    if co.loc[pat].HOSPITAL_EXPIRE_FLAG==1:
          pat_df=pat_df.truncate(after=co.loc[pat].death_time)

    pat_df=pat_df.ffill().dropna()
    # print(pat_df.shape)

    constants=torch.FloatTensor(co.loc[pat][['age','is_male','Weight']]).to(device)
    # constants=torch.FloatTensor(co.loc[pat][['age','is_male']]).to(device)

    T=self.T_max
    if pat_df.shape[0]>T:
          k=np.random.choice(pat_df.shape[0]-T)
          pat_df=pat_df.iloc[k:k+T,:]

    
    treatments=torch.FloatTensor(pat_df[['vaso_rate','volume']].values).to(device)     
    trajectory=torch.FloatTensor(pat_df.drop(['vaso_rate','volume'],axis=1).values).to(device)
    Obs=torch.FloatTensor(pat_df[['HeartRate','SysBP','DiasBP',	'MeanBP']].values).to(device)
    # Obs=torch.FloatTensor(pat_df[['HeartRate','SysBP','DiasBP',	'MeanBP','SpO2','TempC']].values).to(device)         
    
    Others=torch.FloatTensor(pat_df.drop(['SpO2','TempC','liver_24hours','cardiovascular_24hours',
                                      'cns_24hours','renal_24hours','SOFA_24hours'],axis=1).values).to(device)


    return trajectory,treatments, constants, Obs, Others

def collate_model(batch_data):

  """
  We will be a list of tuples,
  len(list) will be batch_size
  (trajectory,treatments, constants, Obs, Others) for each patient in batch
  """
  trajectories=[]
  treatments_list=[]
  seq_lens=[]
  constants_list=[]
  dead_=[]
  Observations=[]
  others_list=[]


  for (trajectory,treatments, constants, Obs, Others) in batch_data:

    trajectories.append(trajectory) 
    treatments_list.append(treatments)
    seq_lens.append(trajectory.shape[0])
    constants_list.append(constants.unsqueeze(0))
    Observations.append(Obs)
    others_list.append(Others)


  
  
  padded_trajectories=pad_sequence(trajectories,batch_first=True)
  padded_treatments=pad_sequence(treatments_list,batch_first=True)
  constants=torch.cat(constants_list)
  mask=get_mini_batch_mask(padded_trajectories,seq_lens)
  padded_obs=pad_sequence(Observations,batch_first=True)
  padded_others=pad_sequence(others_list,batch_first=True)
  

  return padded_trajectories,padded_treatments,constants,mask,seq_lens,padded_obs,padded_others
