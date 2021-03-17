import random
import numpy as np
import torch 
import torch.optim as optim
from .networks import Actor, Critic
from .ofenet import OFENet
import torch.nn.functional as F



class REDQ_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                 state_size,
                 action_size,
                 replay_buffer,
                 ofenet,
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
        self.device = device
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
        
        # split state and action ~ weird step but to keep critic inputs consistent
        self.ofenet = ofenet
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
                    ofenet_loss = self.ofenet.train_ofenet(self.memory.sample())
                experiences = self.memory.sample()
                actor_loss, critic1_loss = self.learn(update, experiences)
        return ofenet_loss, actor_loss, critic1_loss # future ofenet_loss
    
    def act(self, state):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(self.device)
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
        state = torch.from_numpy(state).float().to(self.device)
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
                next_state_features = self.ofenet.get_state_features(next_states).detach()
            else:
                next_state_features = next_states
            next_action, next_log_prob, _ = self.actor_local.sample(next_state_features)
            if self.use_ofenet: 
                next_state_action_features = self.ofenet.get_state_action_features(next_states, next_action).detach() #get_state_action_features
            else:
                next_state_action_features = torch.cat((next_states, next_action), dim=1)
            # TODO: make this variable for possible more than tnext_state_action_featureswo target critics
            Q_target1_next = self.target_critics[idx[0]](next_state_action_features)
            Q_target2_next = self.target_critics[idx[1]](next_state_action_features)
            assert not next_state_features.requires_grad, "next_state_features have gradient but shouldnt!!"
            # take the min of both critics for updating
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(next_action.device) * next_log_prob

        Q_targets = 5.0 * rewards.cpu() + (self.gamma * (1 - dones.cpu()) * Q_target_next.cpu()) # 5.0* (reward_scale)

        # Compute critic losses and update critics 
        if self.use_ofenet:
            state_action_features = self.ofenet.get_state_action_features(states, actions).detach()
        else:
            state_action_features = torch.cat((states, actions), dim=1)
        assert not state_action_features.requires_grad, "State_action_features have gradients but shouldnt!"
        for critic, optim, target in zip(self.critics, self.optims, self.target_critics):
            Q = critic(state_action_features).cpu()
            Q_loss = 0.5 * F.mse_loss(Q, Q_targets)
        
            # Update critic
            optim.zero_grad()
            Q_loss.backward()
            optim.step()    
            # soft update of the targets
            self.soft_update(critic, target)
        
        # ---------------------------- update actor ---------------------------- #
        if step == self.G-1:
            if self.use_ofenet:
                state_features = self.ofenet.get_state_features(states).detach()
            else:
                state_features = states
            assert not state_features.requires_grad, "state features have gradients but shouldnt!"
            actions_pred, log_prob, _ = self.actor_local.sample(state_features)
            
            if self.use_ofenet:
                state_action_features = self.ofenet.get_state_action_features(states, actions_pred)
            else:
                state_action_features = torch.cat((states, actions_pred), dim=1)
            assert state_action_features.requires_grad, "state_action_features should have gradients!"
            # TODO: make this variable for possible more than two critics

            Q1 = self.critics[idx[0]](state_action_features)
            Q2 = self.critics[idx[1]](state_action_features)
            Q = torch.min(Q1,Q2).cpu()
             
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