import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, state_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.states_array = np.empty((buffer_size, state_size))
        self.next_states_array = np.empty((buffer_size, state_size))
        self.actions_array = np.empty((buffer_size, action_size))
        self.dones_array = np.empty((buffer_size, 1))
        self.rewards_array = np.empty((buffer_size, 1))
        self.n_samples = 0
        self.device = device


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        isinstance(state, np.ndarray)
        isinstance(next_state, np.ndarray)
        isinstance(action, np.ndarray)
        isinstance(reward, np.ndarray)
        isinstance(done, np.ndarray)
        
        self.states_array[self.n_samples, ...] = state
        self.next_states_array[self.n_samples, ...] = next_state
        self.actions_array[self.n_samples, ...] = action
        self.rewards_array[self.n_samples, ...] = reward
        self.dones_array[self.n_samples, ...] = done
        self.n_samples += 1
        
    
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.randint(low=0, high=self.n_samples, size=self.batch_size)

        states = torch.tensor(self.states_array[idxs], dtype=torch.float, device=self.device)
        next_states = torch.tensor(self.next_states_array[idxs], dtype=torch.float, device=self.device)
        actions = torch.tensor(self.actions_array[idxs], dtype=torch.float, device=self.device)
        rewards = torch.tensor(self.rewards_array[idxs], dtype=torch.float, device=self.device)
        dones = torch.tensor(self.dones_array[idxs], dtype=torch.float, device=self.device)
        

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.n_samples
    