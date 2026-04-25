import torch
import torch.nn as nn

class SingleFrameCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SingleFrameCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 计算经过卷积和池化后的尺寸
        # 原始尺寸 (45, 60) -> 经过三次池化后变为 (45/8, 60/8) ≈ (5, 7)
        self.fc1 = nn.Linear(128 * 5 * 7, 256)  # 调整为实际尺寸
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 128 * 5 * 7)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 添加dropout
        x = self.fc2(x)
        return torch.sigmoid(x)