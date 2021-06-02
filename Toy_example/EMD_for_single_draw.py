# Calculate expected EMD for the toy example for a single draw
import numpy as np
from scipy import stats

strategy = 1  # 1: all the mass in the central bin (median over outcomes),
              # 2: uniformly distributed over bins (mean over outcomes)
print("Strategy:", strategy)

# Iterate over the number of bins
n_bins_max = 17
for n_bins in range(1, n_bins_max, 2):
    print("Bins:", n_bins)

    # Define the guess depending on the strategy
    if strategy == 1:
        guess_density = np.asarray(n_bins // 2 * [0] + [1] + (n_bins // 2) * [0])
    elif strategy == 2:
        guess_density = np.ones(n_bins) / n_bins
    else:
        raise NotImplementedError

    # Compute cumulative guess
    guess_cum = np.cumsum(guess_density)

    dist_manual = []
    dist_sp = []

    # Calculate the EMD for each possible histogram after a single draw
    for i in range(n_bins):
        correct_density = np.asarray(i * [0] + [1] + (n_bins - i - 1) * [0])  # histogram for outcome Y = i
        correct_cum = np.cumsum(correct_density)

        # Calculate EMD manually as the mean L1 distance between the cumulative histograms
        dist_manual.append(np.abs(guess_cum - correct_cum).mean())

        # Calculate EMD using scipy
        u_and_v_values = np.arange(n_bins) / n_bins
        dist_sp.append(stats.wasserstein_distance(u_and_v_values, u_and_v_values, correct_density, guess_density))

    print("  EMD manual :", np.mean(dist_manual))  # calculate mean over all possible histograms
    print("  EMD scipy  :", np.mean(dist_sp))  # calculate mean over all possible histograms

    if strategy == 1:
        theoretical = (n_bins ** 2 - 1) / (4 * n_bins ** 2)
    else:
        theoretical = (n_bins ** 2 - 1) / (3 * n_bins ** 2)
    print("  Theoretical:", theoretical)
