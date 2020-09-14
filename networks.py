from torch import nn
import torch

class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(17 * 17, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 17 * 17))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DAE(nn.Module):

    def __init__(self):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(17 * 17, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 17 * 17))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CLF(nn.Module):

    def __init__(self):
        super(CLF, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(17 * 17, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            )


    def forward(self, x):
        x = self.model(x)
        return x


