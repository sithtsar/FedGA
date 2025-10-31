#!/usr/bin/env python3
"""
Script to plot extrapolated timing data from benchmark results.
Extrapolates from rounds 11-20 to full 1-50 rounds.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark (rounds 11-20)
rounds = np.arange(11, 21)
baseline_times = np.array(
    [42.50, 39.99, 46.11, 46.26, 41.61, 44.13, 43.68, 40.75, 37.75, 32.64]
)
fedcsga_times = np.array(
    [35.91, 36.17, 36.86, 37.38, 37.34, 37.44, 36.77, 36.30, 35.95, 36.24]
)
genfed_times = np.array(
    [44.67, 47.38, 47.59, 47.98, 51.30, 46.44, 48.91, 44.89, 51.61, 56.27]
)


# Fit linear trends for extrapolation
def extrapolate_times(times, start_round=1, end_round=50):
    # Fit linear regression: time = slope * round + intercept
    slope, intercept = np.polyfit(rounds, times, 1)
    full_rounds = np.arange(start_round, end_round + 1)
    extrapolated = slope * full_rounds + intercept
    # Ensure non-negative times
    extrapolated = np.maximum(extrapolated, 0)
    return full_rounds, extrapolated


# Extrapolate each method
full_rounds, baseline_extrap = extrapolate_times(baseline_times)
_, fedcsga_extrap = extrapolate_times(fedcsga_times)
_, genfed_extrap = extrapolate_times(genfed_times)

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Accuracy placeholder (since we don't have full data, show extrapolated times as main plot)
ax1.plot(full_rounds, baseline_extrap, label="FedAvg Baseline", color="blue")
ax1.plot(full_rounds, fedcsga_extrap, label="FedCSGA", color="orange")
ax1.plot(full_rounds, genfed_extrap, label="GenFed", color="green")
# Highlight actual data points
ax1.scatter(rounds, baseline_times, color="blue", s=20)
ax1.scatter(rounds, fedcsga_times, color="orange", s=20)
ax1.scatter(rounds, genfed_times, color="green", s=20)
ax1.set_xlabel("Rounds")
ax1.set_ylabel("Time (seconds)")
ax1.set_title("Extrapolated Time per Method per Round")
ax1.legend()
ax1.grid(True)

# Time ratios over rounds
baseline_avg = np.mean(baseline_times)
fedcsga_avg = np.mean(fedcsga_times)
genfed_avg = np.mean(genfed_times)

ratios_fedcsga = fedcsga_extrap / baseline_extrap
ratios_genfed = genfed_extrap / baseline_extrap

ax2.plot(full_rounds, ratios_fedcsga, label="FedCSGA/FedAvg", color="orange")
ax2.plot(full_rounds, ratios_genfed, label="GenFed/FedAvg", color="green")
ax2.axhline(y=fedcsga_avg / baseline_avg, color="orange", linestyle="--", alpha=0.7)
ax2.axhline(y=genfed_avg / baseline_avg, color="green", linestyle="--", alpha=0.7)
ax2.set_xlabel("Rounds")
ax2.set_ylabel("Time Ratio")
ax2.set_title("Time Ratios (Extrapolated)")
ax2.legend()
ax2.grid(True)

# Cumulative time
cum_baseline = np.cumsum(baseline_extrap)
cum_fedcsga = np.cumsum(fedcsga_extrap)
cum_genfed = np.cumsum(genfed_extrap)

ax3.plot(full_rounds, cum_baseline, label="FedAvg Baseline", color="blue")
ax3.plot(full_rounds, cum_fedcsga, label="FedCSGA", color="orange")
ax3.plot(full_rounds, cum_genfed, label="GenFed", color="green")
ax3.set_xlabel("Rounds")
ax3.set_ylabel("Cumulative Time (seconds)")
ax3.set_title("Cumulative Time (Extrapolated)")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig("extrapolated_timing.png", dpi=300, bbox_inches="tight")
plt.show()

print("Extrapolated timing plot saved as 'extrapolated_timing.png'")
print(".2f")
print(".2f")
print(".2f")
