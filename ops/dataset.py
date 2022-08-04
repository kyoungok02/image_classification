# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def load_dataset(data_dir, data_type='STL10'):
    if data_type == 'STL10':
        train_ds = datasets.STL10(data_dir, split='train', download=True, transform=transforms.ToTensor())
        val_ds = datasets.STL10(data_dir, split='test', download=True, transform=transforms.ToTensor())
    elif data_type == 'MNIST':
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        val_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    
    # define the image transformation
    transformer = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(224),
                            # transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
                            
        ])
        
    train_ds.transform = transformer
    val_ds.transform = transformer
    
    return train_ds, val_ds

