import torch.nn as nn

class SingleFrameCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.regressor = None

    def forward(self, x):
        feat = self.features(x)
        if self.regressor is None:
            b, c, h, w = feat.shape
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(c * h * w, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 2),
                nn.Sigmoid()
            ).to(x.device)
        return self.regressor(feat)