import torch
import torch.nn as nn
from torch.distributions import Normal


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


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

        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def sample(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(mu.device)
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

        x = torch.relu(self.fc1(state_action))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    
class DenseNetBlock(nn.Module):
    def __init__(self, input_nodes, output_nodes, activation, batch_norm=False, device="cpu"):
        super(DenseNetBlock, self).__init__()
        self.device = device
        self.do_batch_norm = batch_norm
        if batch_norm:
            self.layer = nn.Linear(input_nodes, output_nodes, bias=True).to(device)
            nn.init.xavier_uniform_(self.layer.weight)
            nn.init.zeros_(self.layer.bias)
            self.batch_norm = nn.BatchNorm1d(output_nodes).to(device) #, momentum=0.99, eps=0.001
        else:
            self.layer = nn.Linear(input_nodes, output_nodes).to(device)
            
        if activation == "SiLU":
            self.act = nn.SiLU()
        elif activation == "ReLU":
            self.act = nn.ReLU()
        else:
            print("Activation Function can not be selected!")
    
    def forward(self, x, trainable):

        identity_map = x
        features = self.layer(x)
        # check if this is needed!
        if trainable == False: 
            features = features.detach()
            assert not features.requires_grad
        
        if self.do_batch_norm and trainable:
            features = self.batch_norm(features)
            
        features = self.act(features)
        assert features.shape[0] == identity_map.shape[0], "features: {} | identity: {}".format(features.shape, identity_map.shape)

        features = torch.cat((features, identity_map), dim=1)
        return features
