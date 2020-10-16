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

C1,C2=-0.125,-0.025
def get_rewards(df,dead):
  """
  Get's rewards for a trajectory
  df_sofa: Pandas Df which contains SOFA scores
  MUST BE REINDEXED to have time
  dead (bool): If the terminal is survival or death
  """
  # Calculate rewards for SOFA_{t+1} -SOFA_{t}

  # rewards1=C1*(df.SOFA_24hours-df.shift().SOFA_24hours).dropna().values
  rewards1=C1*(df.SOFA_24hours.values[1:]-df.SOFA_24hours.values[:-1])
  # print(rewards1.shape)
  
  rewards2=C2*((df.shift().SOFA_24hours.iloc[1:]==df.SOFA_24hours.iloc[1:])&(df.SOFA_24hours.iloc[1:]>0)).astype('int').values
  # print(rewards2.shape)
  
  rewards=rewards1+rewards2
  # Calculate Terminal rewards
  if dead:
    rewards=np.concatenate([rewards,[-15]])
  else:
    rewards=np.concatenate([rewards,[15]])

  return rewards


def get_trajectory(pat):

  pat_fluids=input_fluids[input_fluids.icustay_id==pat].set_index('charttime')
  pat_sofa=sofa[sofa.icustay_id==pat].set_index('endtime').resample('H').last().ffill()

  
  pat_sofa=pd.concat([pat_sofa,pat_fluids]).resample('H').sum()[['liver_24hours','cardiovascular_24hours',
                                      'cns_24hours','renal_24hours','SOFA_24hours','Vaso','volume']]

  pat_vitals=vitals[vitals.icustay_id==pat].set_index('charttime')[['HeartRate','SysBP','DiasBP',	'MeanBP','RespRate','SpO2','TempC']]
  

  pat_lab=labs[labs.ICUSTAY_ID==pat]
  pat_lab=pat_lab[['charttime','ANIONGAP','BICARBONATE',	'CREATININE',	'CHLORIDE',	'GLUCOSE',	'HEMATOCRIT',	
           'HEMOGLOBIN','PLATELET'	,'POTASSIUM',	'SODIUM',	'BUN'	,'WBC']]

  pat_lab=pat_lab.set_index('charttime').resample('12H').mean().ffill().dropna()

  if pat_lab.shape[0]<1:
    return None
  
  
  
  labs_with_h=get_lab_hs(pat_lab)



  pat_df=pd.concat([pat_vitals,
                              pat_sofa,labs_with_h]).resample('H').last().ffill().dropna()

 
  dead=co.loc[pat].HOSPITAL_EXPIRE_FLAG==1
  pat_df=pat_df.truncate(before=co.loc[pat].intime-pd.Timedelta('1hr'))

  if co.loc[pat].HOSPITAL_EXPIRE_FLAG==1:
       pat_df=pat_df.truncate(after=co.loc[pat].death_time)

  if pat_df.shape[0]<2:
    return None

  treatments=torch.FloatTensor(pat_df[['Vaso','volume']].values).to(device).unsqueeze(0)     

  trajectory=torch.FloatTensor(pat_df[['HeartRate','SysBP','DiasBP',	'MeanBP','RespRate','SpO2','TempC',
                                      'liver_24hours','cardiovascular_24hours',
                                      'cns_24hours','renal_24hours','SOFA_24hours']].values).to(device).unsqueeze(0)


  obs=torch.FloatTensor(pat_df[['HeartRate','SysBP','DiasBP',	'MeanBP']].values).to(device).unsqueeze(0)
  constants=torch.FloatTensor(co.loc[pat][['age','is_male','Weight']]).to(device).unsqueeze(0)
  
  seq_len=[pat_df.shape[0]]

  with torch.no_grad():
    _,cvs=model(trajectory,treatments,constants,seq_len,obs)

  full_states=pd.DataFrame(np.concatenate([pat_df.values,cvs[:,:,1:].squeeze(0).to('cpu').numpy()],axis=1),
                         index=pat_df.index,columns=list(pat_df.columns)+['R','C','T','SV'])

  full_states['age']=co.loc[pat].age
  full_states['gender']=co.loc[pat].is_male
  full_states['weight']=co.loc[pat].Weight

  treatments=full_states[['Vaso','volume']].values
  rewards=get_rewards(pat_df,dead)
  full_states=full_states.drop(['Vaso','volume'],axis=1)
  # print(full_states)
  return full_states,treatments,rewards


