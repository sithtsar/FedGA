import copy
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from .data_loader import load_mnist
from .fl_base import evaluate, fedavg_aggregate, genfed_aggregate, get_rho_t, train_local
from .ga_selection import ga_client_selection
from .model import create_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    logging.info("Starting FL GA Implementation Simulation")
    # Parameters
    num_clients = 10
    k = 5
    rounds = 50
    alpha = 0.5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(
        f"Parameters: num_clients={num_clients}, k={k}, rounds={rounds}, alpha={alpha}, device={device}",
    )

    # Load data
    logging.info("Loading MNIST data...")
    client_datasets, test_dataset = load_mnist(num_clients, alpha)
    logging.info(
        f"Data loaded: {len(client_datasets)} client datasets, {len(test_dataset)} test samples",
    )

    # Initialize global models for each method
    global_model_base = create_model()
    global_model_fedcsga = create_model()
    global_model_genfed = create_model()

    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)

    # Precompute local accuracies for GA (using initial global model)
    logging.info("Precomputing local accuracies for GA...")
    local_accs = []
    for i in range(num_clients):
        local_model = copy.deepcopy(global_model_fedcsga)
        trained = train_local(local_model, client_datasets[i], device=device)
        acc, _ = evaluate(trained, client_datasets[i], device)
        local_accs.append(acc)
    logging.info(f"Local accuracies computed: {local_accs}")

    # Records
    baseline_acc = []
    baseline_loss = []
    fedcsga_acc = []
    fedcsga_loss = []
    genfed_acc = []
    genfed_loss = []
    round_times = []
    baseline_selections = []
    fedcsga_selections = []
    genfed_selections = []

    for round_num in range(rounds):
        start_time = time.time()
        logging.info(f"Starting Round {round_num + 1}/{rounds}")

        # Baseline: Random selection + FedAvg
        selected = np.random.choice(num_clients, k, replace=False)
        baseline_selections.append(selected.tolist())
        logging.info(f"FedAvg Baseline: Selected clients {selected}")
        local_models = [
            train_local(
                copy.deepcopy(global_model_base), client_datasets[i], device=device,
            )
            for i in selected
        ]
        global_model_base = fedavg_aggregate(
            local_models, [client_datasets[i] for i in selected], device,
        )
        acc, loss = evaluate(global_model_base, test_dataset, device)
        baseline_acc.append(acc)
        baseline_loss.append(loss)
        logging.info(f"FedAvg Baseline: Accuracy {acc:.4f}, Loss {loss:.4f}")
        # Checkpoint
        if (round_num + 1) % 10 == 0 or round_num == rounds - 1:
            torch.save(global_model_base.state_dict(), f"checkpoints/round_{round_num+1}_baseline.pth")

        # FedCSGA: GA selection + FedAvg
        selected = ga_client_selection(num_clients, k, local_accs=local_accs)
        fedcsga_selections.append(selected)
        logging.info(f"FedCSGA (GA Selection + FedAvg): Selected clients {selected}")
        local_models = [
            train_local(
                copy.deepcopy(global_model_fedcsga), client_datasets[i], device=device,
            )
            for i in selected
        ]
        global_model_fedcsga = fedavg_aggregate(
            local_models, [client_datasets[i] for i in selected], device,
        )
        acc, loss = evaluate(global_model_fedcsga, test_dataset, device)
        fedcsga_acc.append(acc)
        fedcsga_loss.append(loss)
        logging.info(f"FedCSGA: Accuracy {acc:.4f}, Loss {loss:.4f}")
        # Checkpoint
        if (round_num + 1) % 10 == 0 or round_num == rounds - 1:
            torch.save(global_model_fedcsga.state_dict(), f"checkpoints/round_{round_num+1}_fedcsga.pth")

        # GenFed: Random selection + GenFed aggregation
        selected = np.random.choice(num_clients, k, replace=False)
        genfed_selections.append(selected.tolist())
        logging.info(f"GenFed (Random Selection + GenFed): Selected clients {selected}")
        local_models = [
            train_local(
                copy.deepcopy(global_model_genfed), client_datasets[i], device=device,
            )
            for i in selected
        ]
        rho_t = get_rho_t(round_num, rounds, "constant", k)
        logging.info(f"GenFed: rho_t = {rho_t}")
        global_model_genfed = genfed_aggregate(
            local_models,
            [client_datasets[i] for i in selected],
            test_dataset,
            rho_t,
            device,
        )
        acc, loss = evaluate(global_model_genfed, test_dataset, device)
        genfed_acc.append(acc)
        genfed_loss.append(loss)
        logging.info(f"GenFed: Accuracy {acc:.4f}, Loss {loss:.4f}")
        # Checkpoint
        if (round_num + 1) % 10 == 0 or round_num == rounds - 1:
            torch.save(global_model_genfed.state_dict(), f"checkpoints/round_{round_num+1}_genfed.pth")

        round_time = time.time() - start_time
        round_times.append(round_time)
        logging.info(f"Round {round_num + 1} completed in {round_time:.2f} seconds")

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    ax1.plot(range(1, rounds + 1), baseline_acc, label="FedAvg Baseline")
    ax1.plot(range(1, rounds + 1), fedcsga_acc, label="FedCSGA")
    ax1.plot(range(1, rounds + 1), genfed_acc, label="GenFed")
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Global Accuracy")
    ax1.set_title("Global Accuracy over Rounds")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(1, rounds + 1), baseline_loss, label="FedAvg Baseline")
    ax2.plot(range(1, rounds + 1), fedcsga_loss, label="FedCSGA")
    ax2.plot(range(1, rounds + 1), genfed_loss, label="GenFed")
    ax2.set_xlabel("Rounds")
    ax2.set_ylabel("Global Loss")
    ax2.set_title("Global Loss over Rounds")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(range(1, rounds + 1), round_times, label="Round Time")
    ax3.set_xlabel("Rounds")
    ax3.set_ylabel("Time (seconds)")
    ax3.set_title("Time per Round")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("results.png")
    plt.show()

    # Plot client selection distributions
    fig2, axs = plt.subplots(1, 3, figsize=(15, 5))
    methods = ["FedAvg Baseline", "FedCSGA", "GenFed"]
    selections_lists = [baseline_selections, fedcsga_selections, genfed_selections]
    for i, (method, sels) in enumerate(zip(methods, selections_lists)):
        counts = [sum(1 for sel in sels if c in sel) for c in range(num_clients)]
        axs[i].bar(range(num_clients), counts)
        axs[i].set_xlabel("Client ID")
        axs[i].set_ylabel("Selection Count")
        axs[i].set_title(f"{method} Client Selections")
        axs[i].grid(True)
    plt.tight_layout()
    plt.savefig("selections.png")
    plt.show()

    # Print final results
    logging.info("Simulation completed")
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
    main()
