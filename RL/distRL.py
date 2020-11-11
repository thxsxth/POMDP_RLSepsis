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
print('Done')


class DistributionalDQN(nn.Module):
    def __init__(self, state_dim, n_actions, N_ATOMS):
        super(DistributionalDQN, self).__init__()

        self.input_layer=nn.Linear(state_dim,512)
        self.n_atoms=N_ATOMS
        self.hiddens=nn.ModuleList([ nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU()) for _ in range(5) ])
        
       
        self.out=nn.Linear(512, n_actions * N_ATOMS)
       
  

    def forward(self, state):
        batch_size =state.size()[0]
        out=self.input_layer(state)

        for layer in self.hiddens:
          out=layer(out)
        
        out=self.out(out)
        
        return F.softmax(out.view(batch_size, -1, self.n_atoms),dim=2),F.log_softmax(out.view(batch_size, -1,self.n_atoms), dim = -1)
class dist_DQN(object):
  def __init__(self,
               num_actions=9,
               state_dim=41,
               v_max=20,
               v_min=-20,
               device='cpu',
               gamma=0.999,
               tau=0.005,
               n_atoms=51
               ):
    self.device=device
    self.Q =DistributionalDQN(state_dim, num_actions,n_atoms).to(self.device)
    self.Q_target=copy.deepcopy(self.Q)
    self.optimizer=torch.optim.Adam(self.Q.parameters(),lr=1e-5)

    self.tau = tau
    self.gamma=gamma
    self.v_min=v_min
    self.v_max=v_max

    self.num_actions = num_actions
    self.atoms=n_atoms
    self.supports = torch.linspace(self.v_min, self.v_max, self.atoms).view(1, 1, self.atoms).to(self.device)
    self.delta = (self.v_max - self.v_min) / (self.atoms - 1)


#     self.z_dist = torch.from_numpy(np.array([[self.v_min + i*self.delta for i in range(self.atoms)]]*batch_size)).to(device)
#     self.z_dist = torch.unsqueeze(self.z_dist, 2).float()



  def train_epoch(self,data_loader,it=0):

    sum_q_loss=0
    for i,(state,next_state,action,reward,done) in enumerate(data_loader):

      batch_size=state.shape[0]
      action=torch.LongTensor(action.reshape(batch_size,1)).to(device)
      reward=torch.FloatTensor(reward.reshape(batch_size,1).cpu().numpy()).to(device)
      done=done.reshape(batch_size,1).to(device)

      batch=(state,next_state,action,reward,done)
      loss=self.compute_loss(batch)
      # print(loss.item())
      sum_q_loss+=loss.item()
       
	    
      if i%25==0:
        print('Epoch :',it,'Batch :', i,'Average Loss :',sum_q_loss/(i+1))


      self.optimizer.zero_grad()
      loss.backward()

      self.optimizer.step()
      self.polyak_target_update()


  def polyak_target_update(self):
     for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


  def compute_loss(self,batch):

     state,next_state,action,reward,done=batch  
     batch_size=state.shape[0]
     range_batch = torch.arange(batch_size).long().to(device)
     action = action.long()

     z_dist = torch.from_numpy(np.array([[self.v_min + i*self.delta for i in range(self.atoms)]]*batch_size)).to(device)
     z_dist = torch.unsqueeze(z_dist, 2).float() 

     _, log_Q_dist_prediction = self.Q(state)
     log_Q_dist_prediction = log_Q_dist_prediction[range_batch, action.squeeze(1), :]

     with torch.no_grad():
        Q_dist_target, _ = self.Q_target(next_state)
 
     Q_target = torch.matmul(Q_dist_target, z_dist).squeeze(1)
     a_star = torch.argmax(Q_target, dim=1)
     Q_dist_star = Q_dist_target[range_batch, a_star.squeeze(1),:]

     m = torch.zeros(batch_size,self.atoms).to(device)

     for j in range(self.atoms):
         T_zj = torch.clamp(reward + self.gamma * (1-done) * (self.v_min + j*self.delta), min = self.v_min, max = self.v_max)
            
         bj = (T_zj - self.v_min)/self.delta
         l = bj.floor().long()
         u = bj.ceil().long()

         mask_Q_l = torch.zeros(m.size()).to(device)
         mask_Q_l.scatter_(1, l, Q_dist_star[:,j].unsqueeze(1))
            
         mask_Q_u = torch.zeros(m.size()).to(device)
         mask_Q_u.scatter_(1, u, Q_dist_star[:,j].unsqueeze(1))
         m += mask_Q_l*(u.float() + (l == u).float()-bj.float())
         m += mask_Q_u*(-l.float()+bj.float())

     loss = - torch.sum(torch.sum(torch.mul(log_Q_dist_prediction, m),-1),-1) / batch_size
     return loss


  def get_action(self,state):
     with torch.no_grad():
      batch_size=state.shape[0]

      z_dist = torch.from_numpy(np.array([[self.v_min + i*self.delta for i in range(self.atoms)]]*batch_size)).to(device)
      z_dist = torch.unsqueeze(z_dist, 2).float()

      Q_dist, _ = self.Q(state)
 
      Q_exp = torch.matmul(Q_dist, z_dist).squeeze(1)
      a_star = torch.argmax(Q_exp, dim=1)

    return a_star


  def get_exp_vals(self,state):
    batch_size=state.shape[0]

    Q_dist, _ = model.Q(states)
    
    z_dist = torch.from_numpy(np.array([[self.v_min + i*self.delta for i in range(self.atoms)]]*batch_size)).to(device)
    z_dist = torch.unsqueeze(z_dist, 2).float()
 
    Q_exp = torch.matmul(Q_dist, self.z_dist).squeeze(1)
	
    return Q_exp


  




