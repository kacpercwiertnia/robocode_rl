import torch.nn as nn

class ShootingNet(nn.Module):
    def __init__(self):
        super(ShootingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
