# Definição da CNN (simples, para focar nos otimizadores)

import torch.nn as nn
import torch.nn.functional as F

class SVHNNet(nn.Module):
    def __init__(self, n_classes=10):
        super(SVHNNet, self).__init__()
        
        # Bloco 1: Entrada 3x32x32 -> Saída 16x16x16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bloco 2: Saída 32x8x8
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Bloco 3: Saída 64x4x4
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Camadas Densas
        self.flatten_dim = 64 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x