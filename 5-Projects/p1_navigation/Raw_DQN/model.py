import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.cnn_layer = Net()
        self.action_value = nn.Linear(256, action_size)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        action_value = self.action_value(x)
        return action_value

    
class Net(nn.Nodule):
    def __init__(self,action_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        return x
