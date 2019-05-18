import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size = [256, 128, 64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"        
        self.fc1 = nn.Linear(state_size, hidden_size[0])
        self.fc1.weight.data.normal_(0, 0.1)    # initialization
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc2.weight.data.normal_(0, 0.1)    # initialization
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc3.weight.data.normal_(0, 0.1)    # initialization
        self.fc4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.fc4.weight.data.normal_(0, 0.1)    # initialization
        self.out = nn.Linear(hidden_size[3], action_size)
        self.out.weight.data.normal_(0, 0.1)    # initialization

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        action_values = self.out(x)
        
        action_fc = F.relu(self.action_fc(x))
        action_value = self.action_value(action_fc)
        # get the state value
        state_value_fc = F.relu(self.state_value_fc(x))
        state_value = self.state_value(state_value_fc)
        # action value mean
        action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
        action_value_center = action_value - action_value_mean
        # Q = V + A
        action_value_out = state_value + action_value_center
        

        return action_values
