import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFeedForwardNN(nn.Module):
    """
    A simple Feedforward Neural Network for tabular features.
    """
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super(SimpleFeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # Using sigmoid for binary classification probability
        return torch.sigmoid(x)

if __name__ == "__main__":
    model = SimpleFeedForwardNN()
    mock_input = torch.randn(32, 10) # Batch of 32, 10 features
    out = model(mock_input)
    print(f"Output shape: {out.shape}")
