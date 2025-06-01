import torch.nn as nn

class ShootingNet(nn.Module):
    def __init__(self, dropout_p: float = 0.5):
        super(ShootingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)
