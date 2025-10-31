# Federated Learning with Genetic Algorithm Client Selection

[![CI](https://github.com/sithtsar/FedGA/actions/workflows/ci.yml/badge.svg)](https://github.com/sithtsar/FedGA/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
1. Clone the repository:
   ```bash
   git clone https://github.com/sithtsar/FedGA.git
   cd FedGA
   ```
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
4. Install pre-commit hooks (optional):
   ```bash
   uv tool install pre-commit
   pre-commit install
   ```

## Usage

### Run Unit Tests
To verify the implementation:
```bash
source .venv/bin/activate
python -m pytest tests/
```
Or with uv:
```bash
uv run python -m pytest tests/
```

### Run Linting and Formatting
Check for errors and format code:
```bash
source .venv/bin/activate
ruff check src/ tests/
ruff format src/ tests/
```
Or with uv:
```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Run the Main Simulation
Execute the FL experiment (runs for 50 rounds by default):
```bash
source .venv/bin/activate
python -m src.fl_ga.main
```
Or with uv:
```bash
uv run python -m src.fl_ga.main
```

The simulation will log progress, including selected clients, accuracies, and losses per round. Results are saved as `results.png` with accuracy and loss plots.

### Customize Parameters
Edit `src/fl_ga/main.py` to change:
- `num_clients`: Number of clients (default 10)
- `k`: Clients selected per round (default 5)
- `rounds`: Number of communication rounds (default 50)
- `alpha`: Dirichlet distribution parameter for non-IID (default 0.5)

## Project Structure

```
fl-ga-implementation/
├── src/
│   └── fl_ga/
│       ├── __init__.py
│       ├── data_loader.py    # MNIST loading and non-IID partitioning
│       ├── model.py          # MLP model definition
│       ├── fl_base.py        # Local training, FedAvg, GenFed aggregation, evaluation
│       ├── ga_selection.py   # GA for client selection
│       └── main.py           # Main simulation script with logging
├── tests/
│   ├── __init__.py
│   └── test_fl.py            # Unit tests
├── docs/
│   ├── report.md             # Detailed mathematical report
│   ├── 1.pdf                 # FedCSGA paper PDF
│   ├── 1.md                  # FedCSGA paper summary
│   ├── 2.pdf                 # GenFed paper PDF
│   ├── 2.md                  # GenFed paper summary
│   └── proj_idea.md          # Project idea and abstract
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── .gitignore
```

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
