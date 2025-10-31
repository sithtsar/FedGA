import matplotlib.pyplot as plt
import numpy as np
import os

# Dummy data generation
rounds = 50
num_clients = 10

# Dummy performance data
baseline_acc = np.random.uniform(0.5, 0.9, rounds).tolist()
baseline_loss = np.random.uniform(0.5, 2.0, rounds).tolist()
fedcsga_acc = np.random.uniform(0.6, 0.95, rounds).tolist()
fedcsga_loss = np.random.uniform(0.3, 1.5, rounds).tolist()
genfed_acc = np.random.uniform(0.55, 0.92, rounds).tolist()
genfed_loss = np.random.uniform(0.4, 1.8, rounds).tolist()
round_times = np.random.uniform(1.0, 5.0, rounds).tolist()

# Dummy selections: each round, list of 5 unique client ids
baseline_selections = [np.random.choice(num_clients, 5, replace=False).tolist() for _ in range(rounds)]
fedcsga_selections = [np.random.choice(num_clients, 5, replace=False).tolist() for _ in range(rounds)]
genfed_selections = [np.random.choice(num_clients, 5, replace=False).tolist() for _ in range(rounds)]

# Plot results (performance)
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
print("Saved results.png")

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
print("Saved selections.png")

print("Plotting test completed successfully!")