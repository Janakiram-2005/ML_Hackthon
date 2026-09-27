import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from model import SimpleFeedForwardNN

def train():
    print("Starting training pipeline...")
    
    # Configuration
    data_path = './data/dataset.csv'
    output_dir = './output'
    epochs = 5
    batch_size = 64
    input_dim = 10 # Change according to your actual feature count
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('./data', exist_ok=True)
    
    # Initialize DataLoader and Model
    dataloader = get_dataloader(csv_file=data_path, batch_size=batch_size)
    model = SimpleFeedForwardNN(input_dim=input_dim)
    
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        
        for x, y in dataloader:
            optimizer.zero_grad()
            
            predictions = model(x)
            loss = criterion(predictions, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / max(batches, 1)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print("--> Saved new best model checkpoint!")
            
    print("Training complete.")

if __name__ == "__main__":
    train()
