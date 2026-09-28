import torch
import torch.nn as nn
import torch.nn.functional as F

class FlightDelayMultiTaskNN(nn.Module):
    """
    Deep Neural Network for Multi-Task Flight Delay Prediction:
    1. Delay Classification (Probability of flight delay)
    2. Delay Regression (Predicted delay duration in minutes)
    """
    def __init__(self, input_dim: int):
        super(FlightDelayMultiTaskNN, self).__init__()

        # Shared feature extractor
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Residual Block
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        # Bottleneck Layer
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.25)

        # Head 1: Classification (Delay Risk)
        self.class_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1)
        )

        # Head 2: Regression (Delay Minutes)
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        
        # Skip connection
        res = x
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = x + res

        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        # Outputs
        delay_logits = self.class_head(x)
        delay_prob = torch.sigmoid(delay_logits)
        
        delay_minutes = F.relu(self.reg_head(x)) # Non-negative delay minutes

        return delay_prob, delay_minutes
