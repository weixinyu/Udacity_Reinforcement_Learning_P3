# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network
import random
import numpy as np
from ddpg import Agent
import torch
from collections import deque, namedtuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, state_size, action_size, buffer_size = int(1e6), batch_size = 128, random_seed = 10):
        
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        
        self.maddpg_agent = [Agent(state_size, action_size), 
                             Agent(state_size, action_size)]
        
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)


    def act(self, obs_all_agents):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions


    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            
    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()
            
    def learn(self, experiences):
        """update the critics and actors of all the agents """
        for agent in self.maddpg_agent:
            agent.learn(experiences)
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

            
            
            




