import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Discrete BCQ algorithm implementation in taken from author's implementation with minor modifications 
https://github.com/sfujim/BCQ
"""

class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions,q_layers=4,i_layers=4):
		super(FC_Q, self).__init__()
		self.q1 = nn.Linear(state_dim, 512)
		self.q2 = nn.Linear(512, 512)
		self.q3 = nn.Linear(512, num_actions)
    
		self.i1 = nn.Linear(state_dim,512)
		self.i2 = nn.Linear(512, 512)
		self.i3 = nn.Linear(512, num_actions)		


	def forward(self, state):
		q = F.relu(self.q1(state))
		q = F.relu(self.q2(q))

		i = F.relu(self.i1(state))
		i = F.relu(self.i2(i))
		i = F.relu(self.i3(i))
		return self.q3(q), F.log_softmax(i, dim=1), i
    
    
class discrete_BCQ(object):
  def __init__(
		self, 
		is_atari,
		num_actions,
		state_dim,
		device,
		BCQ_threshold=0.3,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
	):
	
		  self.device = device

		  # Determine network type
		  self.Q = FC_Q(state_dim, num_actions).to(self.device)
		  self.Q_target = copy.deepcopy(self.Q)
		  self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		  self.discount = discount

		  # Target update rule
		  self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		  self.target_update_frequency = target_update_frequency
		  self.tau = tau

		  # Decay for eps
		  self.initial_eps = initial_eps
		  self.end_eps = end_eps
		  self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		  # Evaluation hyper-parameters
		  self.state_shape =(-1, state_dim)
		  self.eval_eps = eval_eps
		  self.num_actions = num_actions

		  # Threshold for "unlikely" actions
		  self.threshold = BCQ_threshold

		  # Number of training iterations
		  self.iterations = 0


  def select_action(self, state, eval=False):
		# Select action according to policy with probability (1-eps)
		# otherwise, select random action
		  if np.random.uniform(0,1) > self.eval_eps:
			  with torch.no_grad():
				  state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
				  q, imt, i = self.Q(state)
				  imt = imt.exp()
         
				  imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
				  # Use large negative number to mask actions from argmax
				  return ((imt * q + (1. - imt) * -1e8).argmax(1))  #int((imt * q + (1. - imt) * -1e8).argmax(1))
					
		  else:
			  return np.random.randint(self.num_actions)
   
  def train_epoch(self,data_loader,it=0):
    sum_q_loss=0
    sum_i_loss=0
    for batch,(state,next_state,action,reward,done) in enumerate(data_loader):
       # Compute the target Q value
      
      with torch.no_grad():
          batch_size=state.shape[0]
          action=torch.LongTensor(action.reshape(batch_size,1)).to(device)
          reward=torch.FloatTensor(reward.reshape(batch_size,1).cpu().numpy()).to(device)
          done=done.reshape(batch_size,1).to(device)
          
          
          q, imt, i = self.Q(next_state)
          imt = imt.exp()
					
          imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

          # Use large negative number to mask actions from argmax
          next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

          q, imt, i = self.Q_target(next_state)
          target_Q = reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

      current_Q, imt, i = self.Q(state)
      current_Q = current_Q.gather(1, action)

      # Compute Q loss
      q_loss = F.smooth_l1_loss(current_Q, target_Q)
      i_loss = F.nll_loss(imt, action.reshape(-1))

      Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()
      sum_q_loss+=q_loss.item()
      sum_i_loss+=i_loss.item()
	 
      if batch%25==0:
        print(imt[0].exp().detach())
        print('Epoch :',it,'Batch :', batch,'Average q Loss :',sum_q_loss/(batch+1))
        print('Epoch :',it,'Batch :', batch,'Average i Loss :',sum_i_loss/(batch+1))


      # Optimize the Q
      self.Q_optimizer.zero_grad()
      Q_loss.backward()
      self.Q_optimizer.step()

		  # Update target network by polyak or full copy every X iterations.
      self.iterations += 1
      self.maybe_update_target()

  def polyak_target_update(self):
      for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		    


  def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
          self.Q_target.load_state_dict(self.Q.state_dict())
			    
