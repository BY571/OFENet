
import torch
import torch.nn as nn
from .networks import DenseNetBlock
   
class DummyRepresentationLearner():
    def __init__(self, state_size, action_size, target_dim, num_layer=4, hidden_size=40, batch_norm=True, activation="SiLU", device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.target_dim = target_dim
        
    def forward(self, state, action):
        return torch.randn((state[0],self.target_dim))
    
    def get_state_features(self, state):
        return state
    
    def get_state_action_features(self, state, action):
        return torch.cat((state, action), dim=1)

    def train_ofenet(self, experiences, optim):
        pass

    def get_action_state_dim(self,):
        return (self.state_size+self.action_size)
    
    def get_state_dim(self,):
        return self.state_size

class OFENet(nn.Module):
    def __init__(self, state_size, action_size, target_dim, num_layer=4, hidden_size=40, batch_norm=True, activation="SiLU", device="cpu"):
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
                                       batch_norm=batch_norm,
                                       device=device)]
            
        self.state_layer_block = nn.ModuleList(state_layer) 
        self.encode_state_out = state_size + (num_layer) * hidden_size
        action_block_input = self.encode_state_out + action_size
        
        for i in range(num_layer):
            action_layer += [denseblock(input_nodes=action_block_input+i*hidden_size,
                                       output_nodes=hidden_size,
                                       activation=activation,
                                       batch_norm=batch_norm,
                                       device=device)]
        self.action_layer_block = nn.ModuleList(action_layer)

        self.pred_layer = nn.Linear((state_size+(2*num_layer)*hidden_size)+action_size, target_dim)
        
    def forward(self, state, action):
        features = state
        for layer in self.state_layer_block:
            features = layer(features, trainable=True)
        features = torch.cat((features, action), dim=1)
        for layer in self.action_layer_block:
            features = layer(features, trainable=True)
        pred = self.pred_layer(features)
        return pred
    
    def get_state_features(self, state):

        for layer in self.state_layer_block:
            state = layer(state, trainable=False)
        return state
    
    def get_state_action_features(self, state, action):
        
        for layer in self.state_layer_block:
            state = layer(state, trainable=False)
        assert not state.requires_grad
        
        action_cat = torch.cat((state, action), dim=1)
        
        for layer in self.action_layer_block:
            action_cat = layer(action_cat, trainable=False)

        return action_cat
    
    def train_ofenet(self, experiences, optim):
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update OFENet ---------------------------- #
        pred = self.forward(states, actions)
        target_states = next_states[:, :self.target_dim]
        ofenet_loss = (pred - target_states).pow(2).mean()
        

        optim.zero_grad()
        ofenet_loss.backward()
        optim.step()
        return ofenet_loss.item()
    
    def get_action_state_dim(self,):
        return (self.state_size+(2*self.num_layer)*self.hidden_size)+self.action_size
    
    def get_state_dim(self,):
        return self.encode_state_out
    