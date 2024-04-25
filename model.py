import torch
import torch.nn as nn
from datetime import datetime
import time
from torch.nn import functional as F

class ImageNeuralNetwork(nn.Module):
    
    def __init__(self, device, learning_rate = 0.1, epochs = 25):
        
        super().__init__()
        
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # VGG16 Architecture
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64, momentum = 0.9),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64, momentum = 0.9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128, momentum = 0.9),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128, momentum = 0.9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256, momentum = 0.9),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256, momentum = 0.9),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256, momentum = 0.9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512, momentum = 0.9),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512, momentum = 0.9),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512, momentum = 0.9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512, momentum = 0.9),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512, momentum = 0.9),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512, momentum = 0.9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            
            nn.Dropout(0.5),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            
            nn.Linear(4096, 3),
            nn.LogSoftmax(dim = 1)

        )

    def forward(self, x):
        
        return self.network(x)
    
    def set_optimiser(self, optimiser):
        
        self.optimiser = optimiser

