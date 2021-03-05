
import numpy as np
import random
from collections import deque
import time

import json
import gym
import pybullet_envs

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, MultivariateNormal

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import wandb

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def timer(start,end, train_type="Training"):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\n{} Time:  {:0>2}:{:0>2}:{:05.2f}".format(train_type, int(hours),int(minutes),seconds))

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def sample(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mu)
        

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, seed, hidden_size=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        #self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state_action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
class DenseNetBlock(nn.Module):
    def __init__(self, input_nodes, output_nodes, activation, batch_norm=False):
        super(DenseNetBlock, self).__init__()
        
        
        self.do_batch_norm = batch_norm
        if batch_norm:
            self.layer = nn.Linear(input_nodes, output_nodes, bias=True)
            nn.init.xavier_uniform_(self.layer.weight)
            nn.init.zeros_(self.layer.bias)
            self.batch_norm = nn.BatchNorm1d(output_nodes) #, momentum=0.99, eps=0.001
        else:
            self.layer = nn.Linear(input_nodes, output_nodes)
        if activation == "SiLU":
            self.act = nn.SiLU()
        elif activation == "ReLU":
            self.act = nn.ReLU()
        else:
            print("Activation Function can not be selected!")
    
    def forward(self, x):
        identity_map = x
        features = self.layer(x)

        if self.do_batch_norm:
            features = self.batch_norm(features)
        features = self.act(features)
        assert features.shape[0] == identity_map.shape[0], "features: {} | identity: {}".format(features.shape, identity_map.shape)
        #print("FEATURES: {} | STATE: {}".format(features.shape, identity_map.shape))
        features = torch.cat((features, identity_map), dim=1)
        return features
    
    
class OFENet(nn.Module):
    def __init__(self, state_size, action_size, target_dim, num_layer=4, hidden_size=40, batch_norm=True, activation="SiLU"):
        super(OFENet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.target_dim = target_dim
        
        denseblock = DenseNetBlock
        state_layer = []
        action_layer = []
        
        for i in range(num_layer):
            state_layer += [denseblock(input_nodes=state_size+i*hidden_size,
                                       output_nodes=hidden_size,
                                       activation=activation,
                                       batch_norm=batch_norm)]
            
        self.state_layer_block = nn.Sequential(*state_layer)
        self.encode_state_out = state_size + (num_layer) * hidden_size
        action_block_input = self.encode_state_out + action_size
        
        for i in range(num_layer):
            action_layer += [denseblock(input_nodes=action_block_input+i*hidden_size,
                                       output_nodes=hidden_size,
                                       activation=activation,
                                       batch_norm=batch_norm)]
        self.action_layer_block = nn.Sequential(*action_layer)

        self.pred_layer = nn.Linear((state_size+(2*num_layer)*hidden_size)+action_size, target_dim)
        
    def forward(self, state, action):
        features = state
        features = self.state_layer_block(features)
        features = torch.cat((features, action), dim=1)
        features = self.action_layer_block(features)
        pred = self.pred_layer(features)
        return pred
    
    def get_state_features(self, state):
        self.state_layer_block.eval()
        with torch.no_grad():
            z0 = self.state_layer_block(state)
        self.state_layer_block.train()
        return z0
    
    def get_state_action_features(self, state, action):
        self.state_layer_block.eval()
        self.action_layer_block.eval()
        with torch.no_grad():
            z0 = self.state_layer_block(state)
            action_cat = torch.cat((z0, action), dim=1)
            z0_a = self.action_layer_block(action_cat)
        self.state_layer_block.train()
        self.action_layer_block.train()
        return z0_a
    
    def train_ofenet(self, experiences, optim):
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update OFENet ---------------------------- #
        pred = self.forward(states, actions)
        target_states = next_states[:, :self.target_dim]
        ofenet_loss = (target_states - pred).pow(2).mean()
        

        optim.zero_grad()
        ofenet_loss.backward()
        optim.step()
        return ofenet_loss.item()
    
    def get_action_state_dim(self,):
        return (self.state_size+(2*self.num_layer)*self.hidden_size)+self.action_size
    
    def get_state_dim(self,):
        return self.encode_state_out
    

def fill_buffer(agent, env, samples=1000):
    collected_samples = 0
    
    state = env.reset()
    state = state.reshape((1, state_size))
    for i in range(samples):
            
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        next_state = next_state.reshape((1, state_size))
        agent.memory.add(state, action, reward, next_state, done)
        collected_samples += 1
        state = next_state
        if done:
            state = env.reset()
            state = state.reshape((1, state_size))
    print("Adding random samples to buffer done! Buffer size: ", agent.memory.__len__())
                
def pretrain_ofenet(agent, epochs, writer, target_dim):
    losses = []

    for ep in range(epochs):
        states, actions, rewards, next_states, dones = agent.memory.sample()
        # ---------------------------- update OFENet ---------------------------- #
        pred = agent.ofenet.forward(states, actions)
        targets = next_states[:,:target_dim]
        ofenet_loss = (targets-pred).pow(2).mean()
        agent.ofenet_optim.zero_grad()
        ofenet_loss.backward()
        agent.ofenet_optim.step()
        writer.add_scalar("OFENet-pretrainig-loss", ofenet_loss.item(), ep)
    return agent

class REDQ_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                 state_size,
                 action_size,
                 replay_buffer,
                 ofenet=True,
                 target_dim=17,
                 ofenet_layer=8,
                 batch_norm=True,
                 activation="SiLU",
                 lr=3e-4,
                 hidden_size=401,
                 random_seed=0,
                 device="cpu",
                 action_prior="uniform",
                 gamma=0.99,
                 tau=0.005,
                 N=2,
                 M=2,
                 G=1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        feature_size = state_size
        self.action_size = action_size
        feature_action_size = feature_size+action_size
        self.seed = random.seed(random_seed)
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.tau = tau
        self.use_ofenet = ofenet
        
        self.target_entropy = -action_size  # -dim(A)
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=lr) 
        self._action_prior = action_prior
        self.alphas = []
        print("Using: ", device)
        
        # REDQ parameter
        self.N = N # number of critics in the ensemble
        self.M = M # number of target critics that are randomly selected
        self.G = G # Updates per step ~ UTD-ratio
        
        if ofenet:
            ofenet_size = 30
            self.ofenet = OFENet(state_size,
                                action_size,
                                target_dim=target_dim,
                                num_layer=ofenet_layer,
                                hidden_size=ofenet_size,
                                batch_norm=batch_norm,
                                activation=activation).to(device)
            # TODO: CHECK ADAM PARAMS WITH TF AND PAPER
            self.ofenet_optim = optim.Adam(self.ofenet.parameters(), lr=3e-4)  
            print(self.ofenet)

            # split state and action ~ weird step but to keep critic inputs consistent
            feature_size = self.ofenet.get_state_dim()
            feature_action_size = self.ofenet.get_action_state_dim()
        
        # Actor Network 
        self.actor_local = Actor(feature_size, action_size, random_seed, hidden_size=self.hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr)     
        
        # Critic Network (w/ Target Network)
        self.critics = []
        self.target_critics = []
        self.optims = []
        for i in range(self.N):
            critic = Critic(feature_action_size, i, hidden_size=self.hidden_size).to(device)

            optimizer = optim.Adam(critic.parameters(), lr=lr, weight_decay=0)
            self.optims.append(optimizer)
            self.critics.append(critic)
            target = Critic(feature_action_size, i, hidden_size=self.hidden_size).to(device)
            self.target_critics.append(target)


        # Replay memory
        self.memory = replay_buffer
        

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        actor_loss, critic1_loss, ofenet_loss = 0, 0, 0
        for update in range(self.G):
            if len(self.memory) > self.memory.batch_size:
                if self.use_ofenet:
                    ofenet_loss = self.ofenet.train_ofenet(self.memory.sample(), self.ofenet_optim)
                experiences = self.memory.sample()
                actor_loss, critic1_loss = self.learn(update, experiences)
        return ofenet_loss, actor_loss, critic1_loss # future ofenet_loss
    
    def act(self, state):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()

        with torch.no_grad():
            if self.use_ofenet: 
                self.ofenet.eval()
                state = self.ofenet.get_state_features(state)
            action, _, _ = self.actor_local.sample(state)
        self.actor_local.train()
        if self.use_ofenet: self.ofenet.train()
        return action.detach().cpu()[0]
    
    def eval_(self, state):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        
        with torch.no_grad():
            if self.use_ofenet: 
                self.ofenet.eval()
                state = self.ofenet.get_state_features(state)
            _, _ , action = self.actor_local.sample(state)
        self.actor_local.train()
        if self.use_ofenet: self.ofenet.train()
        return action.detach().cpu()[0]
    
    def learn(self, step, experiences):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # sample target critics
        idx = np.random.choice(len(self.critics), self.M, replace=False) # replace=False so that not picking the same idx twice
        

        # ---------------------------- update critic ---------------------------- #

        with torch.no_grad():
            # Get predicted next-state actions and Q values from target models
            if self.use_ofenet:
                next_state_features = self.ofenet.get_state_features(next_states)
            else:
                next_state_features = next_states
            next_action, next_log_prob, _ = self.actor_local.sample(next_state_features)
            if self.use_ofenet: 
                next_state_action_features = self.ofenet.get_state_action_features(next_states, next_action) #get_state_action_features
            else:
                next_state_action_features = torch.cat((next_states, next_action), dim=1)
            # TODO: make this variable for possible more than tnext_state_action_featureswo target critics
            Q_target1_next = self.target_critics[idx[0]](next_state_action_features)
            Q_target2_next = self.target_critics[idx[1]](next_state_action_features)
            
            # take the min of both critics for updating
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(device) * next_log_prob

        Q_targets = 5.0*rewards.cpu() + (self.gamma * (1 - dones.cpu()) * Q_target_next.cpu())

        # Compute critic losses and update critics 
        if self.use_ofenet:
            state_action_features = self.ofenet.get_state_action_features(states, actions)
        else:
            state_action_features = torch.cat((states, actions), dim=1)
        for critic, optim, target in zip(self.critics, self.optims, self.target_critics):
            Q = critic(state_action_features).cpu()
            Q_loss = 0.5*F.mse_loss(Q, Q_targets)
        
            # Update critic
            optim.zero_grad()
            Q_loss.backward()
            optim.step()
            # soft update of the targets
            self.soft_update(critic, target)
        
        # ---------------------------- update actor ---------------------------- #
        if step == self.G-1:
            if self.use_ofenet:
                state_features = self.ofenet.get_state_features(states)
            else:
                state_features = states
            actions_pred, log_prob, _ = self.actor_local.sample(state_features)             
            
            if self.use_ofenet:
                state_action_features = self.ofenet.get_state_action_features(states, actions_pred)
            else:
                state_action_features = torch.cat((states, actions_pred), dim=1)
            # TODO: make this variable for possible more than two critics
            Q1 = self.critics[idx[0]](state_action_features).cpu()
            Q2 = self.critics[idx[0]](state_action_features).cpu()
            Q = torch.min(Q1,Q2)
            
            actor_loss = (self.alpha * log_prob.cpu() - Q).mean()
            # Optimize the actor loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Compute alpha loss 
            alpha_loss = - (self.log_alpha.exp() * (log_prob.cpu() + self.target_entropy).detach().cpu()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()
            self.alphas.append(self.alpha.detach())
            
        return actor_loss.item(), Q_loss.item()

    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

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
    
    
def evaluate(frame, eval_runs=5, capture=False):
    """
    Makes an evaluation run with the current epsilon
    """

    reward_batch = []
    for i in range(eval_runs):
        state = eval_env.reset()

        rewards = 0
        while True:
            action = agent.eval_(np.expand_dims(state, axis=0))
            action_v = np.clip(action, action_low, action_high)
            state, reward, done, _ = eval_env.step(action_v)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    if capture == False:   
        writer.add_scalar("Test_Reward", np.mean(reward_batch), frame)

    
def train(steps, precollected, agent):
    scores_deque = deque(maxlen=100)
    average_100_scores = []
    scores = []
    losses = []

    state = env.reset()
    state = state.reshape((1, state_size))
    score = 0
    i_episode = 1
    for step in range(precollected+1, steps+1):

        # eval runs
        if step % args.eval_every == 0 or step == precollected+1:
            evaluate(step, args.eval_runs)

        action = agent.act(state)
        action_v = action.numpy()
        action_v = np.clip(action_v, action_low, action_high)
        next_state, reward, done, info = env.step(action_v)
        next_state = next_state.reshape((1, state_size))
        ofenet_loss, a_loss, c_loss = agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            scores_deque.append(score)
            scores.append(score)
            average_100_scores.append(np.mean(scores_deque))
            current_step = step - precollected
            writer.add_scalar("Average100", np.mean(scores_deque), current_step)
            writer.add_scalar("Train_Reward", score, current_step)
            writer.add_scalar("OFENet loss", ofenet_loss, current_step)
            writer.add_scalar("Actor loss", a_loss, current_step)
            writer.add_scalar("Critic loss", c_loss, current_step)
            print('\rEpisode {} Frame: [{}/{}] Reward: {:.2f}  Average100 Score: {:.2f} ofenet_loss: {:.3f}, a_loss: {:.3f}, c_loss: {:.3f}'.format(i_episode, step, steps, score, np.mean(scores_deque), ofenet_loss, a_loss, c_loss))
            state = env.reset()
            state = state.reshape((1, state_size))
            score = 0
            i_episode += 1
             

    return scores
        
    
parser = argparse.ArgumentParser(description="")
parser.add_argument("--env", type=str, default="HalfCheetahBulletEnv-v0",
                    help="Environment name, default = HalfCheetahBulletEnv-v0")
parser.add_argument("--info", type=str, default="SAC-OFENet",
                    help="Information or name of the run")
parser.add_argument("--steps", type=int, default=1_000_000,
                    help="The amount of training interactions with the environment, default is 1mio")
parser.add_argument("--N", type=int, default=2,
                    help="Number of Q-network ensemble, default is 10")
parser.add_argument("--M", type=int, default=2,
                    help="Numbe of subsample set of the emsemble for updating the agent, default is 2 (currently only supports 2!)")
parser.add_argument("--G", type=int, default=1,
                    help="Update-to-Data (UTD) ratio, updates taken per step with the environment, default=20")
parser.add_argument("--eval_every", type=int, default=1000,
                    help="Number of interactions after which the evaluation runs are performed, default = 1000")
parser.add_argument("--eval_runs", type=int, default=3,
                    help="Number of evaluation runs performed, default = 1")
parser.add_argument("--seed", type=int, default=0,
                    help="Seed for the env and torch network weights, default is 0")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="Actor learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("--layer_size", type=int, default=401,
                    help="Number of nodes per neural network layer, default is 401")
parser.add_argument("--replay_memory", type=int, default=int(1e6),
                    help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=256,
                    help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=0.005,
                    help="Softupdate factor tau, default is 0.005")
parser.add_argument("-g", "--gamma", type=float, default=0.99,
                    help="discount factor gamma, default is 0.99")
parser.add_argument("--target_dim", type=int, default=17,
                    help="Number of dimension the OFENet has to predict from the state, (default: for Halfcheetah its 17) add")
parser.add_argument("--ofenet_layer", type=int, default=8,
                    help="Number of dense layer in each (state/action) block of the ofenet network, (default: 8)")
parser.add_argument("--collect_random", type=int, default=10_000,
                    help="Number of randomly collected transitions to pretrain the OFENet, (default: 10.000)")
parser.add_argument("--batch_norm", type=int, default=1, choices=[0,1],
                    help="Add batch norm to the OFENet, default: 1")
parser.add_argument("--activation", type=str, default="SiLU", choices=["SiLU", "ReLU"],
                    help="Type of activation function for the ofenet network, choose between SiLU and ReLU, default: SiLU")
parser.add_argument("--ofenet", type=int, default=1, choices=[0,1], help="Using OFENet feature extractor, default: True")

args = parser.parse_args()
    
    
    
if __name__ == "__main__":
        
    writer = SummaryWriter("runs/"+args.info)
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    action_high = env.action_space.high[0]
    seed = args.seed
    action_low = env.action_space.low[0]
    torch.manual_seed(seed)
    env.seed(seed)
    eval_env.seed(seed+1)
    np.random.seed(seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    replay_buffer = ReplayBuffer(action_size, state_size, args.replay_memory, args.batch_size, seed, device)
    agent = REDQ_Agent(state_size=state_size,
                action_size=action_size,
                replay_buffer=replay_buffer,
                ofenet=args.ofenet,
                target_dim=args.target_dim,
                ofenet_layer=args.ofenet_layer,
                batch_norm=args.batch_norm,
                activation=args.activation,
                random_seed=seed,
                lr=args.lr,
                hidden_size=args.layer_size,
                gamma=args.gamma,
                tau=args.tau,
                device=device,
                action_prior="uniform",
                N=args.N,
                M=args.M,
                G=args.G)

    fill_buffer(samples=args.collect_random,
                agent=agent,
                env=env)
    if args.ofenet:
        t0 = time.time()
        agent = pretrain_ofenet(agent=agent,
                        epochs=args.collect_random,
                        writer=writer,
                        target_dim=args.target_dim)
        t1 = time.time()
        timer(t0, t1, train_type="Pre-Training")

    t0 = time.time()
    final_average100 = train(steps=args.steps,
                             precollected=args.collect_random,
                             agent=agent)
    t1 = time.time()
    env.close()
    timer(t0, t1)
    
    # save parameter
    with open('runs/'+args.info+".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    hparams = vars(args)
    metric = {"final average 100 train reward": final_average100}
    writer.add_hparams(hparams, metric)