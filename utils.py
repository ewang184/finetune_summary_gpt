import torch
from torch.utils.data import Dataset, Subset
import random

def get_random_percentage_subset(dataset, percentage):
    """
    Returns a random subset of the dataset based on a given percentage.

    Parameters:
    - dataset (torch.utils.data.Dataset): The original dataset.
    - percentage (float): The percentage of the dataset to return, between 0 and 100.

    Returns:
    - torch.utils.data.Subset: A subset of the dataset.
    """
    # Ensure percentage is between 0 and 100
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100.")

    # Calculate the number of samples to include in the subset
    num_samples = len(dataset)
    subset_size = int(num_samples * (percentage / 100))

    # Generate random indices for the subset
    random_indices = random.sample(range(num_samples), subset_size)

    # Return the subset using the random indices
    return Subset(dataset, random_indices)

# Example usage
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]