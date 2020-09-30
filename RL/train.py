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


from TD3 import TD3
from BCQ import BCQ

from RL_dataset import *
device='cuda' if torch.cuda.is_available() else 'cpu'

ds=DRL_dataset()
loader=DataLoader(ds,batch_size=100,shuffle=True)

def train(model,data_loader,PATH,num_epochs):

for it in range(1,num_epochs+1):
  model.train_epoch(loader,it)
  
  torch.save({
            'Actor_state_dict': model.actor.state_dict(),
            'Actor_target_state_dict': model.actor_target.state_dict(),

            'Critic_state_dict': model.critic.state_dict(),
            'Critic_target_state_dict': model.critic_target.state_dict(),
            'Actor_optim_state_dict':model.actor_optimizer.state_dict(),
            'Critic_optim_state_dict':model.critic_optimizer.state_dict()
          
            
            }, PATH)