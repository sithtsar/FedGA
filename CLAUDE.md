# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements and compares two genetic algorithm-based methods for client selection in Federated Learning (FL):
- **FedCSGA**: GA-based client selection with adaptive operators + FedAvg aggregation
- **GenFed**: Random client selection + GA-inspired aggregation (select top ρ_t models)
- **Baseline**: Random client selection + FedAvg aggregation

The implementation simulates FL with configurable clients on MNIST dataset, partitioned non-IID using Dirichlet distribution.

## Commands

### Dependency Management
- Install dependencies: `uv sync`
- The project uses `uv` for dependency management (not pip)

### Testing
- Run all tests: `uv run python -m pytest tests/`
- Run single test: `uv run python -m pytest tests/test_fl.py::TestFL::test_data_loading`
- Run specific test file: `uv run python -m pytest tests/test_fl.py`

### Code Quality
- Lint code: `uv run ruff check .`
- Format code: `uv run ruff format .`
- Run pre-commit hooks: `pre-commit run --all-files`

### Running Experiments
- Run with default config: `python run.py`
- Run with custom parameters: `python run.py --num-clients 20 --rounds 100 --lr 0.001`
- Run multiple experiments: `python run.py --multiple`
- View all CLI options: `python run.py --help`
- Run main module directly: `uv run python -m src.fl_ga.main`

## Architecture

### Core Components

1. **Configuration System** (`src/fl_ga/config.py`)
   - Centralized `FLConfig` dataclass for all parameters
   - Command-line argument parsing via `from_args()` classmethod
   - Three parameter categories: FL (num_clients, k, rounds, alpha), Training (epochs, lr, batch_size), and GA (pop_size, generations, adaptive, tournament_size)
   - Device auto-detection (CUDA/CPU) if not specified

2. **Data Loading** (`src/fl_ga/data_loader.py`)
   - Loads MNIST and partitions data non-IID using Dirichlet distribution
   - Returns list of client datasets and test dataset
   - Non-IID controlled by alpha parameter (lower = more non-IID)

3. **Model** (`src/fl_ga/model.py`)
   - Lightweight MLP architecture: 784 → 128 → 64 → 10
   - Created via `create_model()` factory function

4. **FL Base Operations** (`src/fl_ga/fl_base.py`)
   - `train_local()`: Local SGD training on client dataset
   - `fedavg_aggregate()`: Weighted averaging by dataset sizes
   - `genfed_aggregate()`: Select top ρ_t models by validation accuracy, then FedAvg
   - `get_rho_t()`: Dynamic ρ_t scheduling (constant/linear/power strategies)
   - `evaluate()`: Compute accuracy and loss on dataset

5. **GA Client Selection** (`src/fl_ga/ga_selection.py`)
   - `ga_client_selection()`: Main GA function for FedCSGA method
   - Fitness function: Average local accuracy of selected clients
   - Tournament selection with configurable size
   - Single-point crossover with chromosome repair (ensures unique client IDs)
   - Mutation: Replace genes with new unique values
   - Adaptive operators: p_c increases from 0.5→0.9, p_m from 0.02→0.05 over generations
   - Special handling for k=1 case (skips crossover when chromosome length ≤ 1)

6. **Main Simulation** (`src/fl_ga/main.py`)
   - Orchestrates comparison of all three methods (Baseline, FedCSGA, GenFed)
   - Each method maintains separate global model
   - Precomputes local accuracies for GA fitness
   - Tracks accuracy, loss, timing, and client selections per round
   - Saves checkpoints to `checkpoints/` directory
   - Outputs plots to `results/` directory

### Data Flow

1. **Initialization**: Load MNIST → partition non-IID → initialize 3 global models
2. **Per Round**:
   - **Baseline**: Random client selection → train locally → FedAvg
   - **FedCSGA**: GA selects clients using local_accs → train locally → FedAvg
   - **GenFed**: Random selection → train locally → evaluate all → select top ρ_t → FedAvg
3. **Evaluation**: Test all global models → record metrics
4. **Output**: Save checkpoints, generate plots

### Important Implementation Details

- **Chromosome Repair**: GA chromosomes represent client selections as lists of unique IDs. After crossover/mutation, `repair_chromosome()` ensures uniqueness by replacing duplicates.
- **k=1 Edge Case**: When k=1 (single client per round), crossover is skipped since chromosome length is 1.
- **Local Accuracies**: Precomputed once at start using initial global model, used as GA fitness proxy throughout training.
- **Device Management**: Uses `pin_memory=True` for DataLoader when device is CUDA for faster data transfer.
- **Checkpointing**: Saves model states, metrics, and selections per round to resume interrupted experiments.

## Configuration Parameters

### FL Parameters
- `num_clients`: Total number of clients (default: 10)
- `k`: Clients selected per round (default: 5)
- `rounds`: Communication rounds (default: 50)
- `alpha`: Dirichlet parameter for non-IID (default: 0.5, lower = more non-IID)

### Training Parameters
- `epochs`: Local training epochs (default: 5)
- `lr`: Learning rate (default: 0.01)
- `batch_size`: Batch size (default: 32)

### GA Parameters
- `pop_size`: Population size (default: 90)
- `generations`: Number of generations (default: 10)
- `adaptive`: Enable adaptive operators (default: True)
- `tournament_size`: Tournament selection size (default: 3)

### GenFed Parameters
- `rho_max`: Maximum number of models to aggregate (default: 5)
- `strategy`: Rho scheduling ('constant', 'linear', 'exponential', default: 'constant')

## Code Style

- Line length: 88 characters
- Quotes: Double quotes for strings
- Type hints: Use proper types, avoid `any` especially in client-side code
- Imports: Group by standard library → third-party → local, one per line
- Logging: Use `logging` module extensively with structured messages
- Error handling: Comprehensive logging for debugging distributed training
