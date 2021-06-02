"""
Script for the plot creation for the bimodal example
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import colorcet as cc
from scipy.stats import norm
sns.set_style("ticks")
sns.set_context("talk")

plt.rcParams["font.size"] = 14
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rc('mathtext', fontset='dejavuserif')

# Set random seed for reproducibility
np.random.seed(0)

# Set parameters
dtype_np = np.float32
dim_in = 3  # input dimension
dim_out = 10  # output dimension: no. of histogram bins
losses = ['EMPL', 'binwise_max_llh_cum']
all_names = ["Truth", "EMPL", "LLH"]
smoothing = 0.0  # smoothing for pinball_CDF
tau_distribution = "uniform"  # distribution to draw tau's from during training
save_root = "./Models"
plot_folder = "./Plots"
os.makedirs(plot_folder, exist_ok=True)

# Select parameters to load (need to have been saved already in NN_histogram_multimodal.py)
alpha_test = 0.5
noise_level_test = 0.5
param_str = str(alpha_test) + "_" + str(noise_level_test)

# Define bins
bin_centres = np.arange(dim_out)
bin_width = np.unique(np.diff(bin_centres))[0]

# # # # # # # #
# EVALUATION FOR EMPL and Gaussian max. llh loss (applied to cumulative histograms)
n_cols = len(losses) + 1
fig, axs = plt.subplots(1, n_cols, figsize=(6, 2.5), squeeze=False, sharex="row", sharey="none")

for i_loss, loss in enumerate(["truth"] + losses):
    loss_load = losses[0] if loss == "truth" else loss
    histogram_data = np.load(os.path.join(save_root, loss_load, "histogram_data_" + param_str + ".npz"))
    print("Loss:", loss)
    alpha_test = histogram_data["alpha_test"]
    noise_level_test = histogram_data["noise_level_test"]
    print("  alpha:", alpha_test, "noise_level", noise_level_test)
    tau_vec = histogram_data["tau_vec"]
    bin_centres_plot = histogram_data["bin_centres_plot"]
    true_quantiles = histogram_data["true_quantiles"]
    samples_cum = histogram_data["samples_cum"]
    pred_test_cum = histogram_data["pred_test_cum"]

    # Make a plot as a function of tau
    # Get number of quantile levels
    n_taus = len(tau_vec)
    n_sigmas = norm.ppf(tau_vec)
    colors = cc.cm.coolwarm(np.linspace(0, 1, n_taus))[::-1]

    # Set some plot settings
    cum_col_1 = [0.25490196, 0.71372549, 0.76862745, 1]
    cum_col_faint = [0.25490196, 0.71372549, 0.76862745, 0.2]

    sns.set_context("talk")
    sns.set_style("ticks")
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    i_row = 0
    i_col = i_loss
    width = 1.0
    x_test_alphas = histogram_data["x_test_alphas"]
    cum_hist = true_quantiles if loss == "truth" else pred_test_cum

    # Iterate over the taus
    for i_tau in range(n_taus):

        # Plot predicted cumulative histogram
        if i_tau < n_taus - 1:
            # Draw the next section of the cumulative histogram in the right colour
            for i in range(len(bin_centres_plot)):
                # Draw the next section of the cumulative histogram in the right colour
                axs[i_row, i_col].fill_between(x=[bin_centres_plot[i] - width / 2.0, bin_centres_plot[i] + width / 2.0],
                                          y1=cum_hist[i_tau, i], y2=cum_hist[i_tau + 1, i], color=colors[i_tau], lw=0)
                # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                if i_tau == 0 and cum_hist[i_tau, i] > 0.99:
                    axs[i_row, i_col].plot([bin_centres_plot[i] - width / 2.0, bin_centres_plot[i] + width / 2.0], 2 * [1.0],
                                      color=colors[0], lw=2, zorder=3)
                elif i_tau == n_taus - 2 and cum_hist[i_tau, i] < 0.01:
                    axs[i_row, i_col].plot([bin_centres_plot[i] - width / 2.0, bin_centres_plot[i] + width / 2.0], 2 * [0.0],
                                      color=colors[-1], lw=2, zorder=3)

    # Set axes limits
    axs[i_row, i_col].set_ylim([-0.175, 1.175])
    if i_col > 0:
        axs[i_row, i_col].set_yticks([])
    else:
        axs[i_row, i_col].set_yticks(np.linspace(0, 1, 5))
    axs[i_row, i_col].set_xticks(np.arange(1, 11))
    axs[i_row, i_col].tick_params(length=6)
    axs[i_row, i_col].text(0.6, 1.1, all_names[i_loss], ha="left", va="top")
    if i_col == 1:
        axs[i_row, i_col].set_xlabel("Bins", fontsize=14, labelpad=0.5)
    if i_col == 0:
        axs[i_row, i_col].set_ylabel("Cum. histogram", fontsize=14)
plt.show()
plt.tight_layout()
plt.subplots_adjust(wspace=0)
fig.savefig(os.path.join(plot_folder, "histogram_plot_" + param_str + ".pdf"), bbox_inches="tight")


## Calibration plot (plotted together with distribution plot below to get the same axis sizes)
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
ax, ax_bin_cum = axs[1], axs[0]
ax.plot([0, 1], [0, 1], "-", lw=1.4, color="0.0")
markers = ["v", "o"]
mfcs = ["white", "k"]
mss = [5, 4.5]
mews = [1, 1]
alphas_plot = [1, 1]

for i_loss, loss in enumerate(losses):
    loss_load = losses[0] if loss == "truth" else loss
    calibration_data = np.load(os.path.join(save_root, loss_load, "calibration_" + param_str + ".npz"))
    print("Loss", loss)
    cutoff = calibration_data["cutoff"]  # if CDF is < cutoff or > 1 - cutoff: ignore in order not to bias results by irrelevant bins
    tol = calibration_data["tol"]
    print("  Cutoff:", cutoff, "tol", tol)
    beta_vec = calibration_data["beta_vec"]
    n_betas = len(beta_vec)
    taus = np.hstack([0.5 - beta_vec[::-1], 0.5 + beta_vec])
    n_taus = len(taus)
    n_sigmas = norm.ppf(taus)
    beta_range_vec = (taus[::-1] - taus)[:n_betas]
    coverage = calibration_data["coverage"]
    x_test_alphas = calibration_data["x_test_alphas"]
    coverage_over_random_inputs = calibration_data["coverage_over_random_inputs"]
    print("  Computed over random inputs?", coverage_over_random_inputs)

    # Make plot
    ax.plot(beta_range_vec, coverage, "k", marker=markers[i_loss], lw=0, markersize=mss[i_loss], mfc=mfcs[i_loss],
            markeredgewidth=mews[i_loss], label=all_names[1 + i_loss], alpha=alphas_plot[i_loss])

# Adjust plot settings
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
ax.set_xlabel(r"Confidence level $\alpha$", fontsize=14)
if coverage_over_random_inputs:
    ax.set_ylabel(r"Coverage $p_{\mathrm{cov}}(\alpha)$", fontsize=14)
else:
    ax.set_ylabel(r"Coverage $p_{\mathrm{cov}}(\alpha \, | \, X)$", fontsize=14)
ax.set_aspect("equal")
ticks = np.linspace(0, 1, 6)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.tick_params(length=6)

## Plot distribution in a selected bin
markers = ["v", "o"]
mfcs = ["white", "k"]
mecs = ["k", "k"]
mss = [5, 4.5]
mews = [1, 1]
alphas_plot = [1.0, 1.0]
truth_col = "0"
truth_ls = "-"
truth_lw = 1.4

for i_loss, loss in enumerate(losses):
    loss_load = losses[0] if loss == "truth" else loss
    bin_data = np.load(os.path.join(save_root, loss_load, "bin_data_" + param_str + ".npz"))
    print("Loss:", loss)
    plot_points = bin_data["plot_points"]
    pred_test_cum = bin_data["pred_test_cum"]
    bin_plot = bin_data["bin_plot"]
    taus_bin = bin_data["taus_bin"]

    # Plot truth
    if i_loss == 0:
        kde_int = bin_data["kde_int"]
        ax_bin_cum.plot(plot_points, kde_int, lw=truth_lw, color=truth_col, ls="-", label="Truth", zorder=5)

    # Plot NN prediction:
    if i_loss == 1:
        # only plot a subset to avoid cluttering the plot
        n_step = 7
        inds_to_plot = np.arange(0, 201, n_step)
        within_valid_range = np.logical_and(pred_test_cum[inds_to_plot, bin_plot] >= 0.0,
                                            pred_test_cum[inds_to_plot, bin_plot] <= 1.0)
        inds_to_plot = inds_to_plot[within_valid_range]

    else:
        n_step = 12
        break_1 = 99
        break_2 = 103
        inds_to_plot = np.hstack([np.arange(0, break_1, n_step),
                                  np.arange(break_1, break_2, 1),
                                  np.arange(break_2, 201, n_step)])

    ax_bin_cum.plot(pred_test_cum[inds_to_plot, bin_plot], taus_bin[inds_to_plot], lw=0, marker=markers[i_loss],
                    ms=mss[i_loss], markeredgewidth=mews[i_loss], alpha=alphas_plot[i_loss], mfc=mfcs[i_loss],
                    mec=mecs[i_loss], label=all_names[i_loss + 1])

# Adjust plot settings
bin_str = str(bin_plot + 1)
ax_bin_cum.set_xlim([-0.05, 1.05])
ax_bin_cum.set_ylim([-0.05, 1.05])
ax_bin_cum.set_xlabel(r"$M_" + bin_str + r"$", fontsize=14)
ax_bin_cum.set_ylabel(r"$F(M_" + bin_str + r"\, | \, X)$", fontsize=14)
ticks = np.linspace(0, 1, 6)
ax_bin_cum.set_xticks(ticks)
ax_bin_cum.set_yticks(ticks)
ax_bin_cum.set_aspect("equal")
ax_bin_cum.tick_params(length=6)
handles, labels = ax_bin_cum.get_legend_handles_labels()
resort = [0, 1, 2]
ax_bin_cum.legend(np.asarray(handles)[resort], np.asarray(labels)[resort], frameon=True, prop={"size": 12},
                  borderpad=0.3, handletextpad=0.75, labelspacing=0.4, handlelength=1.0)  # Plot legend on the LHS
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(plot_folder, "calibration_and_bin_plot_" + param_str + ".pdf"), bbox_inches="tight")
