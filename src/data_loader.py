import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
import os

class ChunkedCSVDataset(IterableDataset):
    """
    An IterableDataset that reads a large CSV file in chunks to save memory.
    """
    def __init__(self, csv_file, chunksize=10000):
        super().__init__()
        self.csv_file = csv_file
        self.chunksize = chunksize

    def __iter__(self):
        if not os.path.exists(self.csv_file):
            # For testing purposes, generate mock data if the file doesn't exist
            yield torch.randn(10), torch.tensor([1.0])
            return

        # Use pandas chunksize to iterate over the large dataset without loading it all into memory
        for chunk in pd.read_csv(self.csv_file, chunksize=self.chunksize):
            # Assuming 'label' is the target column and the rest are features
            if 'label' in chunk.columns:
                labels = chunk.pop('label').values
            else:
                labels = [0] * len(chunk)
                
            features = chunk.values
            
            for i in range(len(features)):
                x = torch.tensor(features[i], dtype=torch.float32)
                y = torch.tensor([labels[i]], dtype=torch.float32)
                yield x, y

def get_dataloader(csv_file='./data/dataset.csv', batch_size=32, chunksize=10000):
    dataset = ChunkedCSVDataset(csv_file, chunksize=chunksize)
    return DataLoader(dataset, batch_size=batch_size)

if __name__ == "__main__":
    # Test the dataloader with a missing file (should yield mock data)
    loader = get_dataloader(csv_file='missing_file.csv')
    for x, y in loader:
        print(f"Batch X shape: {x.shape}, Batch Y shape: {y.shape}")
        break
