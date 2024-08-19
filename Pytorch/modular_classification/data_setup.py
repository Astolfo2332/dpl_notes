"""Module to import data and set the DataLoaders"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

Workers = os.cpu_count()

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int=Workers):
    """
    Creates a Pytorch Dataloader, giving the test and train dirs

    Args: 
    ------
    train_dir: Path to the training directory.
    test_dir: Path yo the test directory.
    transform: transform compose object.
    batch_size: Number of samples por batch in each DataLoader.
    num_workers: Number of python workers per DataLoader.
    Returns:
    A tuple with the DataLoader objects containing:
    (train_dataloader, test_dataloader, class_names)
    """
    #Use ImageFolder to create the data sets
    train_data = datasets.ImageFolder(test_dir, transform=transform)
    test_data = datasets.ImageFolder(train_dir, transform=transform)
    #Extract the class_names
    class_names = train_data.classes
    #Turn the ImageFolder in DataLoader
    train_dataloader = DataLoader(train_data, batch_size, False
    , num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size, False
    , num_workers=num_workers, pin_memory=True)

    return train_dataloader, test_dataloader, class_names
