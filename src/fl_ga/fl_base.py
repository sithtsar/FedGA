import copy

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_local(model, dataset, epochs=5, lr=0.01, batch_size=32, device="cpu"):
    """
    Train a local model on client's dataset.
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return model


def fedavg_aggregate(models, client_datasets, device="cpu"):
    """
    Aggregate models using FedAvg, weighted by dataset sizes.
    """
    total_samples = sum(len(dataset) for dataset in client_datasets)
    global_state = {}

    for key in models[0].state_dict():
        weighted_sum = sum(
            len(client_datasets[i]) * models[i].state_dict()[key].to(device)
            for i in range(len(models))
        )
        global_state[key] = weighted_sum / total_samples

    global_model = models[0].__class__()
    global_model.load_state_dict(global_state)
    return global_model


def evaluate(model, dataset, device="cpu"):
    """
    Evaluate model on dataset.
    """
    model = model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = correct / len(dataset)
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss


if __name__ == "__main__":
    from data_loader import load_mnist
    from model import create_model

    # Test
    client_datasets, test_dataset = load_mnist(num_clients=10, alpha=0.5)
    global_model = create_model()

    # Simulate one round
    local_models = []
    for i in range(5):  # Select 5 clients
        local_model = copy.deepcopy(global_model)
        trained_model = train_local(local_model, client_datasets[i])
        local_models.append(trained_model)

    new_global = fedavg_aggregate(local_models, [client_datasets[i] for i in range(5)])
    acc, loss = evaluate(new_global, test_dataset)
    print(f"Accuracy: {acc:.4f}, Loss: {loss:.4f}")
import numpy as np


def genfed_aggregate(models, client_datasets, test_dataset, rho_t, device="cpu"):
    """
    GenFed aggregation: select top rho_t models based on validation accuracy.
    Using test_dataset as proxy for validation.
    """
    accs = []
    for model in models:
        acc, _ = evaluate(model, test_dataset, device)
        accs.append(acc)

    top_indices = np.argsort(accs)[-rho_t:]
    selected_models = [models[i] for i in top_indices]
    selected_datasets = [client_datasets[i] for i in top_indices]

    return fedavg_aggregate(selected_models, selected_datasets, device)


def get_rho_t(round_num, max_rounds=100, strategy="constant", rho_max=5):
    """
    Dynamic rho_t for GenFed.
    """
    if strategy == "constant":
        return rho_max
    if strategy == "linear":
        return min(rho_max, int(rho_max * (round_num / max_rounds) + 1))
    if strategy == "power":
        b = 0.9  # example
        return min(rho_max, int(rho_max * (1 - b**round_num) + 1))
    return rho_max
