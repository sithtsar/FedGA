import copy
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_mnist
from fl_base import evaluate, fedavg_aggregate, genfed_aggregate, get_rho_t, train_local
from ga_selection import ga_client_selection
from model import create_model

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
    device = "cpu"  # or 'cuda' if available

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
    global_model_p1 = create_model()
    global_model_p2 = create_model()

    # Precompute local accuracies for GA (using initial global model)
    logging.info("Precomputing local accuracies for GA...")
    local_accs = []
    for i in range(num_clients):
        local_model = copy.deepcopy(global_model_p1)
        trained = train_local(local_model, client_datasets[i], device=device)
        acc, _ = evaluate(trained, client_datasets[i], device)
        local_accs.append(acc)
    logging.info(f"Local accuracies computed: {local_accs}")

    # Records
    baseline_acc = []
    baseline_loss = []
    paper1_acc = []
    paper1_loss = []
    paper2_acc = []
    paper2_loss = []

    for round_num in range(rounds):
        start_time = time.time()
        logging.info(f"Starting Round {round_num + 1}/{rounds}")

        # Baseline: Random selection + FedAvg
        selected = np.random.choice(num_clients, k, replace=False)
        logging.info(f"Baseline: Selected clients {selected}")
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
        logging.info(f"Baseline: Accuracy {acc:.4f}, Loss {loss:.4f}")

        # Paper 1: GA selection + FedAvg
        selected = ga_client_selection(num_clients, k, local_accs=local_accs)
        logging.info(f"Paper 1: GA selected clients {selected}")
        local_models = [
            train_local(
                copy.deepcopy(global_model_p1), client_datasets[i], device=device,
            )
            for i in selected
        ]
        global_model_p1 = fedavg_aggregate(
            local_models, [client_datasets[i] for i in selected], device,
        )
        acc, loss = evaluate(global_model_p1, test_dataset, device)
        paper1_acc.append(acc)
        paper1_loss.append(loss)
        logging.info(f"Paper 1: Accuracy {acc:.4f}, Loss {loss:.4f}")

        # Paper 2: Random selection + GenFed aggregation
        selected = np.random.choice(num_clients, k, replace=False)
        logging.info(f"Paper 2: Selected clients {selected}")
        local_models = [
            train_local(
                copy.deepcopy(global_model_p2), client_datasets[i], device=device,
            )
            for i in selected
        ]
        rho_t = get_rho_t(round_num, rounds, "constant", k)
        logging.info(f"Paper 2: rho_t = {rho_t}")
        global_model_p2 = genfed_aggregate(
            local_models,
            [client_datasets[i] for i in selected],
            test_dataset,
            rho_t,
            device,
        )
        acc, loss = evaluate(global_model_p2, test_dataset, device)
        paper2_acc.append(acc)
        paper2_loss.append(loss)
        logging.info(f"Paper 2: Accuracy {acc:.4f}, Loss {loss:.4f}")

        round_time = time.time() - start_time
        logging.info(f"Round {round_num + 1} completed in {round_time:.2f} seconds")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(range(1, rounds + 1), baseline_acc, label="Baseline (Random + FedAvg)")
    ax1.plot(range(1, rounds + 1), paper1_acc, label="Paper 1 (GA Selection + FedAvg)")
    ax1.plot(range(1, rounds + 1), paper2_acc, label="Paper 2 (Random + GenFed)")
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Global Accuracy")
    ax1.set_title("Global Accuracy over Rounds")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(1, rounds + 1), baseline_loss, label="Baseline (Random + FedAvg)")
    ax2.plot(range(1, rounds + 1), paper1_loss, label="Paper 1 (GA Selection + FedAvg)")
    ax2.plot(range(1, rounds + 1), paper2_loss, label="Paper 2 (Random + GenFed)")
    ax2.set_xlabel("Rounds")
    ax2.set_ylabel("Global Loss")
    ax2.set_title("Global Loss over Rounds")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("results.png")
    plt.show()

    # Print final results
    logging.info("Simulation completed")
    logging.info(
        f"Final Baseline Accuracy: {baseline_acc[-1]:.4f}, Loss: {baseline_loss[-1]:.4f}",
    )
    logging.info(
        f"Final Paper 1 Accuracy: {paper1_acc[-1]:.4f}, Loss: {paper1_loss[-1]:.4f}",
    )
    logging.info(
        f"Final Paper 2 Accuracy: {paper2_acc[-1]:.4f}, Loss: {paper2_loss[-1]:.4f}",
    )


if __name__ == "__main__":
    main()
