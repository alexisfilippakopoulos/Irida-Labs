from torch import nn, flatten

class ServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.fc1 = nn.Linear(2*2*512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv5(x))
        x = self.fc1(flatten(x, 1))
        x = self.fc2(x)
        x = self.fc3(x)
        return x