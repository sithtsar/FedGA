"""
Configuration file for FL-GA Implementation
"""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class FLConfig:
    """Configuration for Federated Learning parameters"""

    # Basic FL parameters
    num_clients: int = 10
    k: int = 5  # Number of clients selected per round
    rounds: int = 50
    alpha: float = 0.5  # Data distribution parameter

    # Training parameters
    epochs: int = 5
    lr: float = 0.01
    batch_size: int = 32

    # GA parameters
    pop_size: int = 90
    generations: int = 10
    adaptive: bool = True
    tournament_size: int = 3

    # GenFed parameters
    rho_max: int = 5
    strategy: str = "constant"

    # Device
    device: Optional[str] = None  # Will be auto-detected if None

    @classmethod
    def from_args(cls) -> 'FLConfig':
        """Create config from command line arguments"""
        parser = argparse.ArgumentParser(description='FL-GA Implementation')

        # Basic FL parameters
        parser.add_argument('--num-clients', type=int, default=10,
                          help='Number of clients')
        parser.add_argument('--k', type=int, default=5,
                          help='Number of clients selected per round')
        parser.add_argument('--rounds', type=int, default=50,
                          help='Number of training rounds')
        parser.add_argument('--alpha', type=float, default=0.5,
                          help='Data distribution parameter')

        # Training parameters
        parser.add_argument('--epochs', type=int, default=5,
                          help='Local training epochs')
        parser.add_argument('--lr', type=float, default=0.01,
                          help='Learning rate')
        parser.add_argument('--batch-size', type=int, default=32,
                          help='Batch size')

        # GA parameters
        parser.add_argument('--pop-size', type=int, default=90,
                          help='GA population size')
        parser.add_argument('--generations', type=int, default=10,
                          help='GA generations')
        parser.add_argument('--no-adaptive', action='store_false', dest='adaptive',
                          help='Disable adaptive GA parameters')
        parser.add_argument('--tournament-size', type=int, default=3,
                          help='GA tournament size')

        # GenFed parameters
        parser.add_argument('--rho-max', type=int, default=5,
                          help='Maximum rho value for GenFed')
        parser.add_argument('--strategy', type=str, default='constant',
                          choices=['constant', 'linear', 'exponential'],
                          help='Rho scheduling strategy')

        # Device
        parser.add_argument('--device', type=str, default=None,
                          help='Device to use (cuda/cpu)')

        args = parser.parse_args()

        return cls(
            num_clients=args.num_clients,
            k=args.k,
            rounds=args.rounds,
            alpha=args.alpha,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            pop_size=args.pop_size,
            generations=args.generations,
            adaptive=args.adaptive,
            tournament_size=args.tournament_size,
            rho_max=args.rho_max,
            strategy=args.strategy,
            device=args.device,
        )


# Default configuration instance
default_config = FLConfig()