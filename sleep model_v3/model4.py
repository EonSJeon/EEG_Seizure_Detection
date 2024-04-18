import torch.nn as nn

class SimpleMLP2(nn.Module):
    def __init__(self):
        super(SimpleMLP2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(270, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Output layer for two classes
        )

    def forward(self, x):
        return self.layers(x)

