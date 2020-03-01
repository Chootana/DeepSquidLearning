import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 


class Net(nn.Module):
    def __init__(self, dim_input):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim_input, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)

        return x 

if __name__ == '__main__':
    # check
    net = Net(10)

    input = torch.Tensor(np.arange(10))
    output = net(input)