#!/usr/bin/env python3
"""
Script to run FL-GA experiments with different configurations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fl_ga.config import FLConfig
from fl_ga.main import main


def run_experiment(config: FLConfig):
    """Run a single experiment with the given configuration."""
    print(f"Running experiment with config: {config}")
    main(config)


def run_multiple_experiments():
    """Run multiple experiments with different configurations."""
    # Experiment 1: Default configuration
    print("=== Experiment 1: Default Configuration ===")
    config1 = FLConfig()
    run_experiment(config1)

    # Experiment 2: More clients, fewer rounds
    print("\n=== Experiment 2: More Clients, Fewer Rounds ===")
    config2 = FLConfig(num_clients=20, k=10, rounds=25)
    run_experiment(config2)

    # Experiment 3: Different GA parameters
    print("\n=== Experiment 3: Larger GA Population ===")
    config3 = FLConfig(pop_size=150, generations=15)
    run_experiment(config3)

    # Experiment 4: Different training parameters
    print("\n=== Experiment 4: Different Training Parameters ===")
    config4 = FLConfig(epochs=10, lr=0.001, batch_size=64)
    run_experiment(config4)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Run default experiment
        config = FLConfig()
        run_experiment(config)
    elif sys.argv[1] == "--multiple":
        # Run multiple experiments
        run_multiple_experiments()
    else:
        # Use command line arguments
        config = FLConfig.from_args()
        run_experiment(config)