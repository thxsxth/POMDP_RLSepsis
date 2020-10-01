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
device='cuda' if torch.cuda.is_available() else 'cpu'


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = self.phi * torch.tanh(self.l3(a))
		return (a + action)
  
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-5, 5)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5) #B*Z_dim

		a = F.relu(self.d1(torch.cat([state, z], 1))) #B*(S_dim+Z_dim)
		a = F.relu(self.d2(a))
    
		return (self.d3(a))  # return F.elu(self.d3(a))
    # return self.d3(a)
    
class BCQ(object):

 def __init__(self,state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
  #  latent_dim = action_dim * 5
   latent_dim=8
   self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
   self.actor_target = copy.deepcopy(self.actor)
   self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

   self.critic = Critic(state_dim, action_dim).to(device)
   self.critic_target = copy.deepcopy(self.critic)
   self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

   self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
   self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

   self.max_action = max_action
   self.action_dim = action_dim
   self.discount = discount
   self.tau = tau
   self.lmbda = lmbda
   self.device = device

 def select_action(self, state):	
  with torch.no_grad():
   state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
   action = self.actor(state, self.vae.decode(state))
   q1 = self.critic.q1(state, action)
   ind = q1.argmax(0)
  return action[ind].cpu().data.numpy().flatten()
  
 def train_epoch(self, data_loader, it=0):
    
  sum_critic_loss=0
  sum_actor_loss=0
  sum_vae_loss=0
  for batch,(states,next_states,actions,reward,done) in enumerate(data_loader):
    # Variational Auto-Encoder Training
    batch_size=states.shape[0]
    reward=reward.to(device).reshape(batch_size,-1)
    done=done.to(device).reshape(batch_size,-1)
    recon, mean, std = self.vae(states, actions)
    
    recon_loss = F.mse_loss(recon, actions)
    KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
    vae_loss = recon_loss + 0.5 * KL_loss

    self.vae_optimizer.zero_grad()
    vae_loss.backward()
    self.vae_optimizer.step()
    sum_vae_loss+=vae_loss.item()
    if batch%25==0:
     print('Epoch : ',it+1,' Batch : ', batch+1,'Average VAE Loss : ',sum_vae_loss/(batch+1))

    # Critic Training
    with torch.no_grad():
      next_states = torch.repeat_interleave(next_states, 10, 0)
      
      # Compute value of perturbed actions sampled from the VAE
      target_Q1, target_Q2 = self.critic_target(next_states, self.actor_target(next_states, self.vae.decode(next_states)))
      

      # Soft Clipped Double Q-learning 
      target_Q = (self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)).to(device)
      
      # Take max over each action sampled from the VAE
      target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
      
      target_Q = (reward + (1-done)* self.discount * target_Q)
      target_Q=torch.FloatTensor(target_Q.cpu().numpy()).to(device)    
    
    current_Q1, current_Q2 = self.critic(states, actions)
    # print(current_Q1.shape,current_Q2.shape,target_Q.shape)
    
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    
    
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    sum_critic_loss+=critic_loss.item()
    if batch%25==0:
     print('Epoch : ',it+1,' Batch : ', batch+1,'Average Critic Loss : ',sum_critic_loss/(batch+1) )

    # Pertubation Model / Action Training
    sampled_actions = self.vae.decode(states)
    perturbed_actions = self.actor(states, sampled_actions)

    # Update through DPG
    actor_loss = -self.critic.q1(states, perturbed_actions).mean()
		 	 
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
     
    sum_actor_loss+=actor_loss.item()
    if batch%25==0:
     print('Epoch : ',it+1,' Batch : ', batch+1,'Average Actor Loss : ',sum_actor_loss/(batch+1) )
     print('Epoch : ',it+1,' Batch : ', batch+1,'Actor Loss : ',actor_loss.item() )


    # Update Target Networks 
    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
       target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



BCQ_agent=BCQ(state_dim=41, action_dim=2, max_action=5, device=device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05)


		





	