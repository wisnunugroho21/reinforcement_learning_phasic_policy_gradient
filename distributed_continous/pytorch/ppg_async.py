import gym
from gym.envs.registration import register
    
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy
import time
import datetime

import ray

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Utils():
    def prepro(self, I):
        I           = I[35:195] # crop
        I           = I[::2,::2, 0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0]   = 1 # everything else (paddles, ball) just set to 1
        X           = I.astype(np.float32).ravel() # Combine items in 1 array 
        return X

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, myDevice = None):
        super(Policy_Model, self).__init__()

        self.device = myDevice if myDevice != None else device
        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
              ).float().to(self.device)

        self.actor_layer = nn.Sequential(
                nn.Linear(128, action_dim),
                nn.Tanh()
              ).float().to(self.device)

        self.critic_layer = nn.Sequential(
                nn.Linear(128, 1)
              ).float().to(self.device)
        
    def forward(self, states):
        x = self.nn_layer(states)
        return self.actor_layer(x), self.critic_layer(x)

class Value_Model(nn.Module):
    def __init__(self, state_dim, action_dim, myDevice = None):
        super(Value_Model, self).__init__()   

        self.device = myDevice if myDevice != None else device
        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
              ).float().to(self.device)
        
    def forward(self, states):
        return self.nn_layer(states)

class PolicyMemory(Dataset):
    def __init__(self):
        self.actions        = [] 
        self.states         = []
        self.rewards        = []
        self.dones          = []     
        self.next_states    = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), np.array(self.next_states[idx], dtype = np.float32)      

    def get_all(self):
        return self.states, self.actions, self.rewards, self.dones, self.next_states        
    
    def save_all(self, states, actions, rewards, dones, next_states):
        self.actions = self.actions + actions
        self.states = self.states + states
        self.rewards = self.rewards + rewards
        self.dones = self.dones + dones
        self.next_states = self.next_states + next_states
    
    def save_eps(self, state, action, reward, done, next_state):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]  

class AuxMemory(Dataset):
    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32)

    def save_all(self, states):
        self.states = self.states + states

    def clear_memory(self):
        del self.states[:]

class Continous():
    def __init__(self, myDevice = None):
        self.device = myDevice if myDevice != None else device

    def sample(self, mean, std):
        distribution    = Normal(mean, std)
        return distribution.sample().float().to(self.device)
        
    def entropy(self, mean, std):
        distribution    = Normal(mean, std)    
        return distribution.entropy().float().to(self.device)
      
    def logprob(self, mean, std, value_data):
        distribution    = Normal(mean, std)
        return distribution.log_prob(value_data).float().to(self.device)

    def kl_divergence(self, mean1, std1, mean2, std2):
        distribution1   = Normal(mean1, std1)
        distribution2   = Normal(mean2, std2)

        return kl_divergence(distribution1, distribution2).float().to(self.device)  

class PolicyFunction():
    def __init__(self, gamma = 0.99, lam = 0.95):
        self.gamma  = gamma
        self.lam    = lam

    def monte_carlo_discounted(self, rewards, dones):
        running_add = 0
        returns     = []        
        
        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + (1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)
            
        return torch.stack(returns)
      
    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value           
        return q_values
      
    def generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae     = 0
        adv     = []     

        delta   = rewards + (1.0 - dones) * self.gamma * next_values - values          
        for step in reversed(range(len(rewards))):
            gae = delta[step] + (1.0 - dones[step]) * self.gamma * self.lam * gae
            adv.insert(0, gae)
            
        return torch.stack(adv)

class TrulyPPO():
    def __init__(self, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma, lam):
        self.policy_kl_range    = policy_kl_range
        self.policy_params      = policy_params
        self.value_clip         = value_clip
        self.vf_loss_coef       = vf_loss_coef
        self.entropy_coef       = entropy_coef

        self.distributions      = Continous()
        self.policy_function    = PolicyFunction(gamma, lam)

    def compute_loss(self, action_mean, action_std, old_action_mean, old_action_std, values, old_values, next_values, actions, rewards, dones):    
        # Don't use old value in backpropagation
        Old_values          = old_values.detach()
        Old_action_mean     = old_action_mean.detach()

        # Getting general advantages estimator and returns
        Advantages      = self.policy_function.generalized_advantage_estimation(values, rewards, next_values, dones)
        Returns         = (Advantages + values).detach()
        Advantages      = ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)).detach() 

        # Finding the ratio (pi_theta / pi_theta__old):      
        logprobs        = self.distributions.logprob(action_mean, action_std, actions)
        Old_logprobs    = self.distributions.logprob(Old_action_mean, old_action_std, actions).detach() 

        # Finding Surrogate Loss
        ratios          = (logprobs - Old_logprobs).exp() # ratios = old_logprobs / logprobs        
        Kl              = self.distributions.kl_divergence(Old_action_mean, old_action_std, action_mean, action_std)

        pg_targets  = torch.where(
            (Kl >= self.policy_kl_range) & (ratios > 1),
            ratios * Advantages - self.policy_params * Kl,
            ratios * Advantages
        )
        pg_loss     = pg_targets.mean()

        # Getting entropy from the action probability 
        dist_entropy    = self.distributions.entropy(action_mean, action_std).mean()

        # Getting Critic loss by using Clipped critic value
        if self.value_clip is None:
            critic_loss   = ((Returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped  = old_values + torch.clamp(values - Old_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
            vf_losses1    = (Returns - values).pow(2) * 0.5 # Mean Squared Error
            vf_losses2    = (Returns - vpredclipped).pow(2) * 0.5 # Mean Squared Error        
            critic_loss   = torch.max(vf_losses1, vf_losses2).mean()                

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss 
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss

class JointAux():
    def __init__(self):
        self.distributions  = Continous()

    def compute_loss(self, action_mean, action_std, old_action_mean, old_action_std, values, Returns):
        # Don't use old value in backpropagation
        Old_action_mean     = old_action_mean.detach()

        # Finding KL Divergence                
        Kl              = self.distributions.kl_divergence(Old_action_mean, old_action_std, action_mean, action_std).mean()
        aux_loss        = ((Returns - values).pow(2) * 0.5).mean()

        return aux_loss + Kl

class Learner():  
    def __init__(self, state_dim, action_dim, is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                 batchsize, PPO_epochs, gamma, lam, learning_rate):        
        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batchsize          = batchsize
        self.PPO_epochs         = PPO_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.std                = torch.ones([1, action_dim]).float().to(device)

        self.policy             = Policy_Model(state_dim, action_dim)
        self.policy_old         = Policy_Model(state_dim, action_dim)
        self.policy_optimizer   = Adam(self.policy.parameters(), lr = learning_rate)

        self.value              = Value_Model(state_dim, action_dim)
        self.value_old          = Value_Model(state_dim, action_dim)
        self.value_optimizer    = Adam(self.value.parameters(), lr = learning_rate)

        self.policy_memory      = PolicyMemory()
        self.policy_loss        = TrulyPPO(policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma, lam)

        self.aux_memory         = AuxMemory()
        self.aux_loss           = JointAux()
         
        self.distributions      = Continous()

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def save_all(self, states, actions, rewards, dones, next_states):
        self.policy_memory.save_all(states, actions, rewards, dones, next_states)   

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states):
        action_mean, _      = self.policy(states)
        values              = self.value(states)
        old_action_mean, _  = self.policy_old(states)
        old_values          = self.value_old(states)
        next_values         = self.value(next_states)

        loss                = self.policy_loss.compute_loss(action_mean, self.std, old_action_mean, self.std, values, old_values, next_values, actions, rewards, dones)

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

    def training_aux(self, states):
        Returns                         = self.value(states).detach()

        action_mean, values             = self.policy(states)
        old_action_mean, _              = self.policy_old(states)

        joint_loss                      = self.aux_loss.compute_loss(action_mean, self.std, old_action_mean, self.std, values, Returns)

        self.policy_optimizer.zero_grad()
        joint_loss.backward()
        self.policy_optimizer.step()

    # Update the model
    def update_ppo(self):
        dataloader  = DataLoader(self.policy_memory, self.batchsize, shuffle = False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):
            for states, actions, rewards, dones, next_states in dataloader:
                self.training_ppo(states.float().to(device), actions.float().to(device), rewards.float().to(device), dones.float().to(device), next_states.float().to(device))

        # Clear the memory
        states, _, _, _, _ = self.policy_memory.get_all()
        self.aux_memory.save_all(states)
        self.policy_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

    def update_aux(self):
        dataloader  = DataLoader(self.aux_memory, self.batchsize, shuffle = False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states in dataloader:
                self.training_aux(states.float().to(device))

        # Clear the memory
        self.aux_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_weights(self):
        torch.save(self.policy.state_dict(), 'agent.pth')

class Agent:  
    def __init__(self, state_dim, action_dim, is_training_mode):
        self.is_training_mode   = is_training_mode
        self.device             = torch.device('cpu')    

        self.memory             = PolicyMemory() 
        self.distributions      = Continous(self.device)
        self.policy             = Policy_Model(state_dim, action_dim, self.device)  
        self.std                = torch.ones([1, action_dim]).float().to(self.device)      
        
        if is_training_mode:
          self.policy.train()
        else:
          self.policy.eval()

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)
    
    def get_all(self):
        return self.memory.get_all()

    def act(self, state):
        state           = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()
        action_mean, _  = self.policy(state)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action  = self.distributions.sample(action_mean, self.std)
        else:
            action  = action_mean  
              
        return action.squeeze(0).cpu().numpy()

    def set_weights(self, weights):
        self.policy.load_state_dict(weights)

    def load_weights(self):
        self.policy.load_state_dict(torch.load('agent.pth', map_location = self.device))

@ray.remote
class Runner():
    def __init__(self, env_name, training_mode, render, n_update, tag):       
        self.utils              = Utils()

        self.env                = gym.make(env_name)
        self.states             = self.env.reset()
        self.state_dim          = self.env.observation_space.shape[0]
        self.action_dim         = self.env.action_space.shape[0]

        self.agent              = Agent(self.state_dim, self.action_dim, training_mode)

        self.render             = render
        self.tag                = tag
        self.training_mode      = training_mode
        self.n_update           = n_update
        self.max_action         = 1.0

    def run_episode(self, i_episode, total_reward, eps_time):
        self.agent.load_weights()

        for _ in range(self.n_update):
            action = self.agent.act(self.states) 

            action_gym = np.clip(action, -1.0, 1.0) * self.max_action
            next_state, reward, done, _ = self.env.step(action_gym)

            eps_time += 1 
            total_reward += reward
            
            if self.training_mode:
                self.agent.save_eps(self.states.tolist(), action, reward, float(done), next_state.tolist())
                
            self.states = next_state
                    
            if self.render:
                self.env.render()

            if done:
                self.states = self.env.reset()
                i_episode   += 1

                print('Episode {} \t t_reward: {} \t time: {} \t process no: {} \t'.format(i_episode, total_reward, eps_time, self.tag))

                total_reward = 0
                eps_time = 0             
        
        return self.agent.get_all(), i_episode, total_reward, eps_time, self.tag

def plot(datas):
    print('----------')

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))

def main():
    ############## Hyperparameters ##############
    training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it

    render              = True # If you want to display the image, set this to True. Turn this off if you run this in Google Collab
    n_update            = 1024 # How many episode before you update the Policy. Recommended set to 1024 for Continous
    n_aux_update        = 5
    n_episode           = 100000 # How many episode you want to run
    n_agent             = 2 # How many agent you want to run asynchronously

    policy_kl_range     = 0.03 # Recommended set to 0.03 for Continous
    policy_params       = 5 # Recommended set to 5 for Continous
    value_clip          = 1.0 # How many value will be clipped. Recommended set to the highest or lowest possible reward
    entropy_coef        = 0.0 # How much randomness of action you will get. Because we use Standard Deviation for Continous, no need to use Entropy for randomness
    vf_loss_coef        = 1.0 # Just set to 1
    batch_size          = 32 # How many batch per update. size of batch = n_update / batch_size. Recommended set to 32 for Continous
    PPO_epochs          = 10 # How many epoch per update. Recommended set to 10 for Continous    
    
    gamma               = 0.99 # Just set to 0.99
    lam                 = 0.95 # Just set to 0.95
    learning_rate       = 2.5e-4 # Just set to 0.95
    #############################################
    env_name            = 'BipedalWalker-v3'

    env                 = gym.make(env_name)
    state_dim           = env.observation_space.shape[0]
    action_dim          = env.action_space.shape[0]

    learner             = Learner(state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                            batch_size, PPO_epochs, gamma, lam, learning_rate)     
    #############################################
    t_aux_updates = 0
    start = time.time()
    ray.init()    

    try:
        runners = [Runner.remote(env_name, training_mode, render, n_update, i) for i in range(n_agent)]
        learner.save_weights()

        episode_ids = []
        for i, runner in enumerate(runners):
            episode_ids.append(runner.run_episode.remote(i, 0, 0))
            time.sleep(4)

        for _ in range(1, n_episode + 1):
            ready, not_ready = ray.wait(episode_ids)
            trajectory, i_episode, total_reward, eps_time, tag = ray.get(ready)[0]

            states, actions, rewards, dones, next_states = trajectory
            learner.save_all(states, actions, rewards, dones, next_states)

            learner.update_ppo()
            t_aux_updates += 1

            if t_aux_updates == n_aux_update:
                learner.update_aux()
                t_aux_updates = 0

            learner.save_weights()

            episode_ids = not_ready
            episode_ids.append(runners[tag].run_episode.remote(i_episode, total_reward, eps_time))

    except KeyboardInterrupt:        
        print('\nTraining has been Shutdown \n')
    finally:
        ray.shutdown()

        finish = time.time()
        timedelta = finish - start
        print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))

if __name__ == '__main__':
    main()