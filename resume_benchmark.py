#!/usr/bin/env python3
"""
Script to resume FL-GA experiments from checkpoints and benchmark timing per method.
Loads checkpoints at round 10 and runs 10 additional rounds.
"""

import copy
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fl_ga.config import FLConfig
from fl_ga.data_loader import load_mnist
from fl_ga.fl_base import (
    evaluate,
    fedavg_aggregate,
    genfed_aggregate,
    get_rho_t,
    train_local,
)
from fl_ga.ga_selection import ga_client_selection
from fl_ga.model import create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def resume_benchmark(
    config: FLConfig = None, checkpoint_round=10, additional_rounds=10
):
    if config is None:
        config = FLConfig()

    # Set device
    if config.device is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Resuming FL GA Benchmark from checkpoints")
    logging.info(f"Loading checkpoints at round {checkpoint_round}")
    logging.info(
        f"Running {additional_rounds} additional rounds (rounds {checkpoint_round + 1} to {checkpoint_round + additional_rounds})"
    )
    logging.info(
        f"Parameters: num_clients={config.num_clients}, k={config.k}, device={config.device}",
    )

    # Load data
    logging.info("Loading MNIST data...")
    client_datasets, test_dataset = load_mnist(config.num_clients, config.alpha)
    logging.info(
        f"Data loaded: {len(client_datasets)} client datasets, {len(test_dataset)} test samples",
    )

    # Load checkpoints
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoints directory '{checkpoint_dir}' not found")

    global_model_base = create_model()
    global_model_fedcsga = create_model()
    global_model_genfed = create_model()

    baseline_path = os.path.join(
        checkpoint_dir, f"round_{checkpoint_round}_baseline.pth"
    )
    fedcsga_path = os.path.join(checkpoint_dir, f"round_{checkpoint_round}_fedcsga.pth")
    genfed_path = os.path.join(checkpoint_dir, f"round_{checkpoint_round}_genfed.pth")

    if not all(os.path.exists(p) for p in [baseline_path, fedcsga_path, genfed_path]):
        raise FileNotFoundError(
            f"One or more checkpoint files missing at round {checkpoint_round}"
        )

    global_model_base.load_state_dict(
        torch.load(baseline_path, map_location=config.device)
    )
    global_model_fedcsga.load_state_dict(
        torch.load(fedcsga_path, map_location=config.device)
    )
    global_model_genfed.load_state_dict(
        torch.load(genfed_path, map_location=config.device)
    )
    logging.info("Checkpoints loaded successfully")

    # Recompute local accuracies for GA using the fedcsga checkpoint model
    logging.info("Recomputing local accuracies for GA using checkpoint model...")
    local_accs = []
    for i in range(config.num_clients):
        local_model = copy.deepcopy(global_model_fedcsga)
        trained = train_local(
            local_model,
            client_datasets[i],
            config.epochs,
            config.lr,
            config.batch_size,
            config.device,
        )
        acc, _ = evaluate(trained, client_datasets[i], config.device)
        local_accs.append(acc)
    logging.info(f"Local accuracies recomputed: {local_accs}")

    # Records for additional rounds
    baseline_acc = []
    baseline_loss = []
    fedcsga_acc = []
    fedcsga_loss = []
    genfed_acc = []
    genfed_loss = []
    baseline_times = []
    fedcsga_times = []
    genfed_times = []

    for round_offset in range(additional_rounds):
        round_num = checkpoint_round + round_offset  # For rho_t calculation
        logging.info(
            f"Starting Round {round_num + 1}/{checkpoint_round + additional_rounds}"
        )

        # Baseline: Random selection + FedAvg
        start_time = time.time()
        selected = np.random.choice(config.num_clients, config.k, replace=False)
        logging.info(f"FedAvg Baseline: Selected clients {selected}")
        local_models = [
            train_local(
                copy.deepcopy(global_model_base),
                client_datasets[i],
                config.epochs,
                config.lr,
                config.batch_size,
                config.device,
            )
            for i in selected
        ]
        global_model_base = fedavg_aggregate(
            local_models,
            [client_datasets[i] for i in selected],
            config.device,
        )
        acc, loss = evaluate(global_model_base, test_dataset, config.device)
        baseline_acc.append(acc)
        baseline_loss.append(loss)
        baseline_time = time.time() - start_time
        baseline_times.append(baseline_time)
        logging.info(
            f"FedAvg Baseline: Accuracy {acc:.4f}, Loss {loss:.4f}, Time {baseline_time:.2f}s"
        )

        # FedCSGA: GA selection + FedAvg
        start_time = time.time()
        selected = ga_client_selection(
            config.num_clients,
            config.k,
            config.pop_size,
            config.generations,
            local_accs=local_accs,
            adaptive=config.adaptive,
            tournament_size=config.tournament_size,
        )
        logging.info(f"FedCSGA (GA Selection + FedAvg): Selected clients {selected}")
        local_models = [
            train_local(
                copy.deepcopy(global_model_fedcsga),
                client_datasets[i],
                config.epochs,
                config.lr,
                config.batch_size,
                config.device,
            )
            for i in selected
        ]
        global_model_fedcsga = fedavg_aggregate(
            local_models,
            [client_datasets[i] for i in selected],
            config.device,
        )
        acc, loss = evaluate(global_model_fedcsga, test_dataset, config.device)
        fedcsga_acc.append(acc)
        fedcsga_loss.append(loss)
        fedcsga_time = time.time() - start_time
        fedcsga_times.append(fedcsga_time)
        logging.info(
            f"FedCSGA: Accuracy {acc:.4f}, Loss {loss:.4f}, Time {fedcsga_time:.2f}s"
        )

        # GenFed: Random selection + GenFed aggregation
        start_time = time.time()
        selected = np.random.choice(config.num_clients, config.k, replace=False)
        logging.info(f"GenFed (Random Selection + GenFed): Selected clients {selected}")
        local_models = [
            train_local(
                copy.deepcopy(global_model_genfed),
                client_datasets[i],
                config.epochs,
                config.lr,
                config.batch_size,
                config.device,
            )
            for i in selected
        ]
        rho_t = get_rho_t(round_num, config.rounds, config.strategy, config.k)
        logging.info(f"GenFed: rho_t = {rho_t}")
        global_model_genfed = genfed_aggregate(
            local_models,
            [client_datasets[i] for i in selected],
            test_dataset,
            rho_t,
            config.device,
        )
        acc, loss = evaluate(global_model_genfed, test_dataset, config.device)
        genfed_acc.append(acc)
        genfed_loss.append(loss)
        genfed_time = time.time() - start_time
        genfed_times.append(genfed_time)
        logging.info(
            f"GenFed: Accuracy {acc:.4f}, Loss {loss:.4f}, Time {genfed_time:.2f}s"
        )

    # Print timing summary
    logging.info("=== Timing Summary ===")
    logging.info(
        f"Average FedAvg Baseline time per round: {np.mean(baseline_times):.2f}s"
    )
    logging.info(f"Average FedCSGA time per round: {np.mean(fedcsga_times):.2f}s")
    logging.info(f"Average GenFed time per round: {np.mean(genfed_times):.2f}s")
    logging.info(
        f"FedCSGA/FedAvg time ratio: {np.mean(fedcsga_times) / np.mean(baseline_times):.2f}"
    )
    logging.info(
        f"GenFed/FedAvg time ratio: {np.mean(genfed_times) / np.mean(baseline_times):.2f}"
    )

    # Print final results
    logging.info("=== Final Results After Resume ===")
    logging.info(
        f"Final FedAvg Baseline Accuracy: {baseline_acc[-1]:.4f}, Loss: {baseline_loss[-1]:.4f}",
    )
    logging.info(
        f"Final FedCSGA Accuracy: {fedcsga_acc[-1]:.4f}, Loss: {fedcsga_loss[-1]:.4f}",
    )
    logging.info(
        f"Final GenFed Accuracy: {genfed_acc[-1]:.4f}, Loss: {genfed_loss[-1]:.4f}",
    )


if __name__ == "__main__":
    config = FLConfig.from_args()
    resume_benchmark(config)
