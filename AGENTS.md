# Agent Guidelines for FL-GA Implementation

## Build/Lint/Test Commands
- **Install dependencies**: `uv sync`
- **Run all tests**: `uv run python -m pytest tests/`
- **Run single test**: `uv run python -m pytest tests/test_fl.py::TestFL::test_data_loading`
- **Lint code**: `uv run ruff check .`
- **Format code**: `uv run ruff format .`

## Configuration System
- **Config file**: `src/fl_ga/config.py` contains all configurable parameters
- **Command line**: Use `python run.py --help` to see all available options
- **Run with defaults**: `python run.py`
- **Run with custom config**: `python run.py --num-clients 20 --rounds 100 --lr 0.001`
- **Run multiple experiments**: `python run.py --multiple`

### Configurable Parameters
- **FL Parameters**: `num_clients`, `k`, `rounds`, `alpha`
- **Training**: `epochs`, `lr`, `batch_size`
- **GA Parameters**: `pop_size`, `generations`, `adaptive`, `tournament_size`
- **GenFed**: `rho_max`, `strategy`
- **Device**: `device` (auto-detected if not specified)

## Code Style Guidelines
- **Line length**: 88 characters (ruff default)
- **Quotes**: Use double quotes for strings
- **Indentation**: 4 spaces
- **Imports**: Standard library → third-party → local imports, one per line
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error handling**: Use logging module extensively with structured messages
- **Docstrings**: Use triple quotes for class/function documentation
- **Types**: Avoid `any` type in client-side code; use proper type hints where applicable
- **ML frameworks**: Use torch/torchvision/torchaudio for deep learning operations

## Project Structure
- Source code in `src/fl_ga/`
- Tests in `tests/`
- Configuration in `src/fl_ga/config.py`
- Runner script in `run.py`
- Use relative imports within the package
- Save model checkpoints to `checkpoints/` directory
- Output plots to root directory (results.png, selections.png)

## Development Workflow
- Run pre-commit hooks: `pre-commit run --all-files`
- Use `logging.info()` for important events and progress tracking
- Test locally before committing changes
- Modify `config.py` to add new configurable parameters