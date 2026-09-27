import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from model import SimpleFeedForwardNN

def test_model_forward_pass():
    model = SimpleFeedForwardNN(input_dim=10)
    # Create mock tensor (batch of 16, 10 features)
    mock_input = torch.randn(16, 10)
    output = model(mock_input)
    
    # Assert output shape matches expected (batch_size, 1)
    assert output.shape == (16, 1), f"Expected shape (16, 1), got {output.shape}"
    
    # Assert outputs are probabilities (between 0 and 1 due to sigmoid)
    assert torch.all((output >= 0) & (output <= 1)), "Outputs should be probabilities between 0 and 1"
