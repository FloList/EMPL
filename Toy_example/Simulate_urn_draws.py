"""
This script compares analytic and numerical results for the toy example.
Scenario:
Consider a discrete uniform distribution from 0 to n-1 (=> p = 1 / n for each integer)
Draw k times from this discrete uniform distribution.
1) The probability of drawing any number j exactly l times is given by
     P(X = l) = (k choose l) p^l (1 - p)^(k - l),
where p = 1 / n.
Note that the _relative_ fraction X / k for k draws has mean p and variance p (1-p) / k -> 0 for k -> infinity.
2) Now, what is the probability of drawing exactly l values that are <= a number y?
     P(l values in k draws are <= y) =
            (k choose l) \hat{p}^l (1 - \hat{p})^(k - l),
where \hat{p} = (1 + floor(y)) / n if y >= 0.
3) Finally, what is the probability of drawing l or less values that are <= a number y?
     P(at most l values in k draws are <= y) =
        sum_{m=0}^{m=l} (k choose m) \hat{p}^m (1 - \hat{p})^(k - m),
where \hat{p} is defined as above.
That this is true can be checked with this script.
4) In order to calculate the quantile function, start from P(at most l values in k draws are <= y), evaluate this for
all l in [0, ..., k], and invert by finding the first index such that F <= tau.
NOTE: in the paper, 1-based indexing is used, whereas this file uses 0-based indexing.
"""
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
sns.set_style("ticks")
sns.set_context("talk")
plt.rcParams["figure.figsize"] = [8, 8]

# Settings
n_sims = 5000
n = 5  # no. bins
p = 1 / n  # probability for each bin
p_hat = lambda y: (np.floor(y) + 1) / n  # probability for a count to fall in a bin less or equal y
k = 3  # no. of drawings
l_step_eval = 1  # step for evaluation of l's
y_log_scale = None  # set log scale for y axis? None: automatic depending on what is plotted

# 1) drawing exactly l times a specific number j:
print("Drawing exactly l times a specific number j (same for all j).")
l = np.arange(0, k + 1, l_step_eval)  # number of times that exactly j is drawn
sol_analyt, sol_numer = [], []
j_to_check = 0
for l_ in l:
    sol_analyt.append(stats.binom.pmf(l_, k, p))
    # Simulate
    drawings = np.random.choice(n, size=(n_sims, k))
    sol_numer.append((np.sum(drawings == j_to_check, 1) == l_).sum() / n_sims)
    print("  l =", l_, "Analytical:", np.round(sol_analyt[-1], 3), "Numerical:", np.round(sol_numer[-1], 3))
sol_analyt, sol_numer = np.asarray(sol_analyt), np.asarray(sol_numer)
fig_1, ax_1 = plt.subplots(1, 1)
ax_1.plot(l, sol_analyt, "ro")
ax_1.plot(l, sol_numer, "bx")
ax_1.set_title("Probability of drawing exactly l times a specific number j")
ax_1.set_xlabel("l")
ax_1.set_ylabel("Prob.")
if y_log_scale in [True, None]:
    ax_1.set_yscale("log")
print("\n\n")


# 2) drawing exactly l times a number less or equal a specific number j:
print("Drawing exactly l times a number less or equal a specific number j.")
j = np.arange(0, n, 1)
l = np.arange(0, k + 1, l_step_eval)  # number of times that a number less or equal j is drawn
drawings = np.random.choice(n, size=(n_sims, k))
sol_analyt, sol_numer = [], []
for j_ in j:
    for l_ in l:
        sol_analyt.append(stats.binom.pmf(l_, k, p_hat(j_)))
        sol_numer.append((np.sum(drawings <= j_, 1) == l_).sum() / n_sims)
        print("j =", j_,  "l =", l_, "Analytical:", np.round(sol_analyt[-1], 3), "Numerical:", np.round(sol_numer[-1], 3))
sol_analyt, sol_numer = np.asarray(sol_analyt), np.asarray(sol_numer)
sol_analyt = np.reshape(sol_analyt, [len(j), -1])
sol_numer = np.reshape(sol_numer, [len(j), -1])
fig_2, axs_2 = plt.subplots(len(j), 1, sharex="all", sharey="all")
for i_ax, ax in enumerate(axs_2.flatten()):
    ax.plot(l, sol_analyt[i_ax], "ro")
    ax.plot(l, sol_numer[i_ax], "bx")
    if i_ax == 0:
        ax.set_title("Exactly l times drawing Y <= j.")
    if i_ax == len(j) - 1:
        ax.set_xlabel("l")
    ax.set_ylabel("j=" + str(j[i_ax]))
    if y_log_scale in [True, None]:
        ax.set_yscale("log")
print("\n\n")

# 3)
# a) drawing at most / at least l times a number less or equal a specific number j:
print("Drawing at most / at least l times a number less or equal a specific number j.")
sol_analyt_atmost, sol_numer_atmost = [], []
sol_analyt_atleast, sol_numer_atleast = [], []
for j_ in j:
    for l_ in l:
        sol_analyt_atmost.append(np.sum([stats.binom.pmf(m, k, p_hat(j_)) for m in range(l_ + 1)]))
        sol_numer_atmost.append((np.sum(drawings <= j_, 1) <= l_).sum() / n_sims)
        sol_analyt_atleast.append(np.sum([stats.binom.pmf(m, k, p_hat(j_)) for m in range(l_, k + 1)]))
        sol_numer_atleast.append((np.sum(drawings <= j_, 1) >= l_).sum() / n_sims)
        # print("j =", j_,  "l =", l_, "Analytical at most:", np.round(sol_analyt_atmost[-1]), "Numerical at most:", np.round(sol_numer_atmost[-1], 3))
        # print("j =", j_,  "l =", l_, "Analytical at least:", np.round(sol_analyt_atleast[-1]), "Numerical at least:", np.round(sol_numer_atleast[-1], 3))
sol_analyt_atmost, sol_numer_atmost = np.asarray(sol_analyt_atmost), np.asarray(sol_numer_atmost)
sol_analyt_atleast, sol_numer_atleast = np.asarray(sol_analyt_atleast), np.asarray(sol_numer_atleast)
sol_analyt_atmost = np.reshape(sol_analyt_atmost, [len(j), -1])
sol_analyt_atleast = np.reshape(sol_analyt_atleast, [len(j), -1])
sol_numer_atmost = np.reshape(sol_numer_atmost, [len(j), -1])
sol_numer_atleast = np.reshape(sol_numer_atleast, [len(j), -1])

fig_3, axs_3 = plt.subplots(len(j), 2, sharex="all", sharey="all")
for i_ax, ax in enumerate(axs_3):
    ax[0].plot(l, sol_analyt_atmost[i_ax], "ro")
    ax[0].plot(l, sol_numer_atmost[i_ax], "bx")
    ax[1].plot(l, sol_analyt_atleast[i_ax], "ro")
    ax[1].plot(l, sol_numer_atleast[i_ax], "bx")
    if i_ax == 0:
        ax[0].set_title("At most l times drawing Y <= j")
        ax[1].set_title("At least l times drawing Y <= j")
    if i_ax == len(j) - 1:
        ax[0].set_xlabel("l")
        ax[1].set_xlabel("l")
    ax[0].set_ylabel("j=" + str(j[i_ax]))
    if y_log_scale:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
print("\n\n")

# Quantile function
colors = cc.cm.bkr(np.linspace(0, 1, n))[::-1]

# Quantiles to calculate
taus_all = [1 / (2 * n) + j_ / n for j_ in range(n)]

# Entry (j, l): probability that at most l times a number less or equal j is drawn
def return_quantile_points(n, k):
    quantile_points_raw = []
    for j_ in range(n):
        quantile_points_raw.append([stats.binom.pmf(m, k, p_hat(j_)) for m in range(k + 1)])
    quantile_points_raw = np.asarray(quantile_points_raw)
    quantile_points = quantile_points_raw.cumsum(1)
    return quantile_points

# Quantiles: analytically
def return_quantiles(quantile_points, taus, n, k):
    # invert: find quantiles
    quantiles_analyt = []
    for tau in taus:
        quantiles_analyt.append(np.asarray([np.argmin(quantile_points[j_, :] <= tau) for j_ in range(n)]) / k)
    return np.asarray(quantiles_analyt)

quantile_points = return_quantile_points(n, k)
quantiles_analyt = return_quantiles(quantile_points, taus_all, n, k)

# Quantiles: numerically
drawings_hist = np.asarray([np.histogram(drawings[i, :], bins=np.arange(n + 1), density=True)[0] for i in range(n_sims)])
drawings_hist_cum = drawings_hist.cumsum(1)
drawings_quantiles = np.asarray([np.quantile(drawings_hist_cum, tau, axis=0) for tau in taus_all])

# Make a plot
fig_quant, ax_quant = plt.subplots(1, 1)
for j_ in range(n):
    ax_quant.step(np.arange(n), drawings_quantiles[j_, :], color=colors[j_], where="post")
    ax_quant.step(np.arange(n), quantiles_analyt[j_, :], color="gold", ls="dotted", alpha=0.5, where="post")
ax_quant.legend(["Numerical", "Analytical"])
ax_quant.set_xlabel("j")
ax_quant.set_ylabel("Prob.")
