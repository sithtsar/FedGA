import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

def dirichlet_partition(dataset, num_clients, alpha=0.5):
    """
    Partition dataset into non-IID subsets using Dirichlet distribution.
    """
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    client_data_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        # Sample proportions using Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)
        proportions = np.insert(proportions, 0, 0)
        
        for i in range(num_clients):
            start, end = proportions[i], proportions[i+1]
            client_data_indices[i].extend(class_indices[start:end])
    
    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(client_data_indices[i])
    
    return client_data_indices

def load_mnist(num_clients=10, alpha=0.5):
    """
    Load MNIST and partition into non-IID client datasets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    client_indices = dirichlet_partition(train_dataset, num_clients, alpha)
    client_datasets = [Subset(train_dataset, indices) for indices in client_indices]
    
    return client_datasets, test_dataset

if __name__ == "__main__":
    # Test the data loading
    client_datasets, test_dataset = load_mnist(num_clients=10, alpha=0.5)
    print(f"Number of clients: {len(client_datasets)}")
    for i, dataset in enumerate(client_datasets):
        print(f"Client {i}: {len(dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
