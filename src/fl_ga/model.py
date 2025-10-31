import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Lightweight MLP for MNIST classification.
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_model():
    """
    Create and return the MLP model.
    """
    return MLP()

if __name__ == "__main__":
    model = create_model()
    print(model)
    # Test with dummy input
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"Output shape: {output.shape}")
