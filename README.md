# Federated Learning with Genetic Algorithm Client Selection

This project implements and compares two genetic algorithm-based methods for client selection in Federated Learning (FL), based on the papers:

1. **FedCSGA**: Optimizing Client Selection in Federated Learning Based on Genetic Algorithm with Adaptive Operators (Wu et al., 2025)
2. **GenFed**: Accelerating Federated Learning with Genetic Algorithm Enhancements (Zheng et al., 2025)

## Project Overview

The implementation simulates FL with 10 clients on MNIST dataset, partitioned non-IID using Dirichlet distribution (alpha=0.5). The model is a lightweight MLP with two hidden layers (128 and 64 units).

### Methods Compared

- **Baseline**: Random client selection + FedAvg aggregation
- **Paper 1 (FedCSGA)**: GA-based client selection with adaptive operators + FedAvg aggregation
- **Paper 2 (GenFed)**: Random client selection + GA-inspired aggregation (select top ρ_t models)

## Installation

### Prerequisites
- Python 3.8+
- uv (for dependency management)

### Setup
1. Clone or navigate to the project directory.
2. Install uv if not already installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```
3. Create virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```
4. Install ruff for linting (optional):
   ```bash
   uv tool install ruff
   ```

## Running the Code

### Run Unit Tests
To verify the implementation:
```bash
source .venv/bin/activate
python tests.py
```
Or with uv:
```bash
uv run python tests.py
```

### Run Linting and Formatting
Check for errors and format code:
```bash
source .venv/bin/activate
ruff check .
ruff format .
```
Or with uv:
```bash
uv run ruff check .
uv run ruff format .
```

### Run the Main Simulation
Execute the FL experiment (runs for 50 rounds by default):
```bash
source .venv/bin/activate
python main.py
```
Or with uv:
```bash
uv run python main.py
```

The simulation will log progress, including selected clients, accuracies, and losses per round. Results are saved as `results.png` with accuracy and loss plots.

### Customize Parameters
Edit `main.py` to change:
- `num_clients`: Number of clients (default 10)
- `k`: Clients selected per round (default 5)
- `rounds`: Number of communication rounds (default 50)
- `alpha`: Dirichlet distribution parameter for non-IID (default 0.5)

## Key Components

- `data_loader.py`: MNIST loading and non-IID partitioning
- `model.py`: MLP model definition
- `fl_base.py`: Local training, FedAvg, GenFed aggregation, evaluation
- `ga_selection.py`: GA for client selection
- `main.py`: Main simulation script with logging
- `tests.py`: Unit tests
- `pyproject.toml`: Project configuration for uv

## Results and Analysis

The simulation generates plots comparing accuracy and loss over rounds for all methods. Logs provide detailed per-round metrics.

Expected outcomes: GA-optimized selection improves accuracy by 5-10% and reduces convergence rounds by 20-30% compared to random selection.

## Reproduction

- Dataset: MNIST
- Clients: 10
- Non-IID: Dirichlet α=0.5
- Model: MLP (784-128-64-10)
- Local epochs: 5
- LR: 0.01
- Batch size: 32
