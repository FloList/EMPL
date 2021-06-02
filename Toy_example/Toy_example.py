"""
Script for the toy example.
NOTE: script is 0-based, while the description in the paper is 1-based (-> bins are shifted by 1!).
Input variable is number of draws, output is the relative frequency histogram.
"""

import os
import keras
from keras import layers
from keras.losses import Loss
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from fractions import Fraction
import seaborn as sns
import colorcet as cc
sns.set_style("ticks")
sns.set_context("talk")
tfd = tfp.distributions

# Set random seed for reproducibility
tf.random.set_seed(0)
np.random.seed(0)

# Set parameters
batch_size = 2048  # batch size (2048)
dim_in = 1  # input dimension (1)
dim_out = 5  # output dimension: no. of histogram bins (5) CHOOSE ODD!
n_hidden = 128  # hidden neurons (128)
n_layers = 2  # number of hidden layers (2)
act_fun = "relu"  # activation function for the hidden layers ("relu")
act_fun_final = "softmax"  # activation function for the output layers ("softmax" for EMPL, "linear" for binwise_max_llh_cum)
do_batch_norm = True  # batch normalisation (True)
n_epochs = 10000  # number of epochs (= batch iterations in this case) for training (10000)
x_min, x_max = 0.0, 1.0  # limits for uniform distribution to draw input from (0, 1)
bias_power = 1.0  # bias the training data generation: > 1 -> more small numbers, < 1 -> more large numbers (1)
loss = "EMPL"  # loss function: mean_absolute_error (bin-wise), x-ent, EM1, or EMPL
smoothing = 0.0  # smoothing for EMPL (0)
tau_distribution = "uniform"  # distribution to draw tau's from during training
save_path = "./Models"
os.makedirs(save_path, exist_ok=True)

# Define mapping for input
x_mapping = lambda xi: (xi - 0.5) * 12  # x |-> x shown to the NN
tau_mapping = lambda t: (t - 0.5) * 12  # tau |-> tau shown to the NN
k_log_max = 3
k_base = 10
k_mapping = lambda ka: k_base ** (k_log_max * ka)   # uniformly drawn variable (except bias power) |-> number of drawings k

# Define data pipeline
class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, batch_size=16, dim_in=1, dim_out=1, generate_tau=False):
        """Initialization"""
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_size = batch_size
        self.on_epoch_end()
        self.generate_tau = generate_tau

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return 1

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate data
        x, y = self.__data_generation()

        # Generate tau and append to input and output
        if self.generate_tau:
            if tau_distribution == "uniform":
                tau = tf.random.uniform((self.batch_size, 1), 0.0, 1.0)
            elif tau_distribution == "arcsin":
                tau = 0.5 + 0.5 * tf.math.sin(tf.random.uniform([self.batch_size, 1], -np.pi, np.pi))
            else:
                raise NotImplementedError

            x = tf.concat([tau_mapping(tau), x], axis=1)  # scale tau for input
            y = tf.concat([tau, y], axis=1)
        return x, y

    def on_epoch_end(self):
        return

    def __data_generation(self):
        """Generates data containing batch_size samples"""
        # Generate data
        x_raw = x_min + tf.random.uniform([self.batch_size, self.dim_in], 0.0, 1.0) ** bias_power * (x_max - x_min)
        y = self.get_label(x_raw)

        # Map x
        x = x_mapping(x_raw)

        # Return
        return x, y

    # k is determined as a function of the input
    @classmethod
    def get_label(cls, x):
        n_batch_x = x.shape[0]
        k_draws = tf.cast(k_mapping(x), tf.int32)
        probs = np.ones((n_batch_x, dim_out), dtype=np.float32) / dim_out
        dist = tfd.Multinomial(total_count=tf.cast(tf.reshape(k_draws, -1), tf.float32), probs=probs)
        dist_draw = dist.sample(1)[0]
        hist = dist_draw / tf.reduce_sum(dist_draw, 1, keepdims=True)

        # Slow approach
        # k_draws_cum = tf.concat([[[0]], tf.cumsum(k_draws)], axis=0)
        # k_draws_tot = k_draws_cum[-1]
        # y_min = 0
        # y_max = dim_out
        # y = y_min + tf.random.uniform(k_draws_tot, 0.0, 1.0, dtype=tf.float32) * (y_max - y_min)
        # y_resh = [y[k_draws_cum[i, 0]:k_draws_cum[i+1, 0]] for i in range(n_batch_x)]  # THIS IS SLOW!
        # hist = tf.reshape(tf.concat([tf.histogram_fixed_width(y_resh[i], nbins=dim_out, value_range=[0.0, dim_out])
        #                              for i in range(len(y_resh))], axis=0), [n_batch_x, -1]) / k_draws
        return tf.cast(hist, tf.float32)


# Loss functions
def emd_loss(p, p_hat, r=1, scope="emd_loss", do_root=True):
    """Compute the Earth Mover's Distance loss."""
    with tf.name_scope(scope):
        ecdf_p = tf.math.cumsum(p, axis=-1)
        ecdf_p_hat = tf.math.cumsum(p_hat, axis=-1)
        if r == 1:
            emd = tf.reduce_mean(tf.abs(ecdf_p - ecdf_p_hat), axis=-1)
        elif r == 2:
            emd = tf.reduce_mean((ecdf_p - ecdf_p_hat) ** 2, axis=-1)
            if do_root:
                emd = tf.sqrt(emd)
        else:
            emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), r), axis=-1)
            if do_root:
                emd = tf.pow(emd, 1 / r)
        return tf.reduce_mean(emd)


class EMPL(Loss):
    """Compute the EMPL"""
    def __init__(self, name="empl", reduction=keras.losses.Reduction.AUTO, smoothing=0.0):
        super().__init__(reduction=reduction, name=name)
        self.smoothing = smoothing
        self.name = name

    def call(self, data, y_pred):
        with tf.name_scope(self.name):
            tau, y_true = data[:, :1], data[:, 1:]
            ecdf_p = tf.math.cumsum(y_true, axis=-1)
            ecdf_p_hat = tf.math.cumsum(y_pred, axis=-1)
            delta = ecdf_p_hat - ecdf_p

            # Non-smooth C0 loss (default)
            if self.smoothing == 0.0:
                mask = tf.cast(tf.greater_equal(delta, tf.zeros_like(delta)), tf.float32) - tau
                loss = mask * delta

            # Smooth loss
            else:
                loss = -tau * delta + self.smoothing * tf.math.softplus(delta / self.smoothing)

            mean_loss = tf.reduce_mean(loss, axis=-1)
            final_loss = tf.reduce_mean(mean_loss)

            return final_loss


# Build generators
generate_tau = loss == "EMPL"
generator = DataGenerator(batch_size=batch_size, dim_in=dim_in, dim_out=dim_out, generate_tau=generate_tau)

# Define custom activation functions
def softplus_norm(x):
    return keras.activations.softplus(x) / K.sum(keras.activations.softplus(x), 1, keepdims=True)

def softplus_power_norm(x, power=0.5):
    return keras.activations.softplus(x) ** power / K.sum(keras.activations.softplus(x) ** power, 1, keepdims=True)

def sigmoid_norm(x):
    return keras.activations.sigmoid(x) / K.sum(keras.activations.sigmoid(x), 1, keepdims=True)

def relu_norm(x):
    return keras.activations.relu(x) / K.sum(keras.activations.relu(x), 1, keepdims=True)

get_custom_objects().update({'softplus_norm': layers.core.Activation(softplus_norm)})
get_custom_objects().update({'softplus_power_norm': layers.core.Activation(softplus_power_norm)})
get_custom_objects().update({'sigmoid_norm': layers.core.Activation(sigmoid_norm)})
get_custom_objects().update({'relu_norm': layers.core.Activation(relu_norm)})

# Build NN
dim_in_concat = dim_in + 1 if loss == "EMPL" else dim_in

model = keras.Sequential()
model.add(layers.Dense(n_hidden, activation=act_fun, input_shape=(dim_in_concat,)))
if do_batch_norm:
    model.add(layers.BatchNormalization())
for i_layer in range(n_layers - 1):
    model.add(layers.Dense(n_hidden, activation=act_fun))
    if do_batch_norm:
        model.add(layers.BatchNormalization())
model.add(layers.Dense(dim_out, activation=act_fun_final))
model.build()
model.summary()

# Select loss
if loss == "mean_absolute_error":
    loss_tf = loss
elif loss == "x-ent":
    loss_tf = keras.losses.CategoricalCrossentropy(from_logits=False)
elif loss == "EM1":
    loss_tf = emd_loss
elif loss == "EMPL":
    loss_tf = EMPL(smoothing=smoothing)
else:
    raise NotImplementedError

# Compile NN
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=n_epochs, decay_rate=0.1, staircase=False)

# Load / train NN
model_path = os.path.join(save_path, loss)

if os.path.exists(model_path):
    model.load_weights(os.path.join(model_path, "weights"))
    model.compile(loss=loss_tf, optimizer='adam', metrics=[])
else:
    model.compile(loss=loss_tf, optimizer='adam', metrics=[])
    os.makedirs(model_path, exist_ok=True)
    history = model.fit(x=generator, epochs=n_epochs, verbose=2)
    model.save_weights(os.path.join(model_path, "weights"))

    # Show training progress
    fig_history, ax_history = plt.subplots(1, 1)
    ax_history.plot(history.history["loss"])
    ax_history.set_xlabel("Epoch")
    ax_history.set_ylabel("Loss")

# Define bins
bin_centres = range(dim_out)
bin_width = np.unique(np.diff(bin_centres))[0]

# # # # # # # #

# 1) Plot for a fixed tau
# Evaluate NN
x_test = np.linspace(0, 1, k_log_max + 1)
k_test = k_mapping(x_test)
n_eval = len(x_test)
x_test = np.expand_dims(x_test, -1)
x_test_mapped = x_mapping(x_test)
tau_eval = 0.5
y_test = generator.get_label(x_test)

# Concatenate tau to input
if loss == "EMPL":
    tau = tau_eval * np.ones((n_eval, 1))
    x_test_feed = np.concatenate([tau_mapping(tau), x_test_mapped], 1)
else:
    x_test_feed = x_test_mapped

# Predict
pred_test = model.predict(x_test_feed)

# Plot guess with random realisation
fig_1, axs_1 = plt.subplots(1, 4, sharex="all", sharey="all", figsize=(8, 2))
for i, ax in enumerate(axs_1.flatten()):
    if i >= n_eval:
        ax.axis("off")
        ax.text(0.5, 0.5, "Loss: " + loss.replace("_", " ").capitalize())
        break
    ax.bar(bin_centres, y_test[i], fc="darkgreen", ec="k", lw=2, alpha=0.5, width=1.0, align="center")
    ax.bar(bin_centres, pred_test[i], fc="firebrick", ec="k", lw=2, alpha=0.5, width=1.0, align="center")
    this_x_test = x_test[i]
    ax.set_title(str(np.round(this_x_test[0], 2)))
    ax.set_xticks(range(dim_out))
    fig_1.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

if loss == "EMPL":
    p_hat = lambda y, n: (np.floor(y) + 1) / n  # probability for a count to fall in a bin less or equal y

    # Entry (j, l): probability that at most l times a number less or equal j is drawn
    def return_quantile_points(n, k):
        quantile_points_raw = []
        for j_ in range(n):
            quantile_points_raw.append([stats.binom.pmf(m, k, p_hat(j_, n)) for m in range(k + 1)])
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

    # Get axis ratio
    def get_aspect(ax):
        from operator import sub
        # Total figure size
        figW, figH = ax.get_figure().get_size_inches()
        # Axis size on figure
        _, _, w, h = ax.get_position().bounds
        # Ratio of display units
        disp_ratio = (figH * h) / (figW * w)
        # Ratio of data units
        # Negative over negative because of the order of subtraction
        data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
        return disp_ratio / data_ratio


    # 2) Make a plot as a function of tau
    # Set up vector of x for evaluation
    x_step = 1
    x_values_frac = [Fraction(j, k_log_max) for j in range(0, k_log_max + 1, x_step)]
    x_values = np.asarray([x_values_frac[i].__float__() for i in range(len(x_values_frac))])
    k_values = k_mapping(x_values).astype(int)
    n_x = len(x_values)

    # Define quantile levels tau
    taus_all_frac = [Fraction(1, (2 * dim_out)) + Fraction(j, dim_out) for j in range(dim_out)]
    taus_all = np.asarray([taus_all_frac[i].__float__() for i in range(len(taus_all_frac))])
    n_taus = len(taus_all)
    colors = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]
    true_col = '#2f4f4fff'
    median_col = "orange"

    # Append bin zero to plot CDF = 0 on the LHS
    bin_centres_ext = [-1] + list(bin_centres) + [dim_out]

    # Plot options
    show_k = True
    alpha_pred = 0.5
    alpha_bins = 0.12

    # Options for CDF plot for central bin
    show_dist = True  # show distribution within central bin?
    x_lim = [-0.25, 1.25]
    y_lim = [-0.12, 1.12]
    centre_ind = (dim_out - 1) // 2
    p_binom = (1 + centre_ind) / dim_out  # p for central bin or less
    dist_col = "white"
    arr_pos_flip = -0.06

    # Settings for arrows in CDF plot
    pad_thick = 1
    arr_length = 0.11111
    arr_width = 0.025
    arr_lw = 0

    # Make figure
    plot_cols = 2 if show_dist else 1
    gridspec_kw = {'width_ratios': [2.5, 1]} if show_dist else {}
    fig_tau, axs_tau = plt.subplots(n_x, plot_cols, sharex="col", sharey="col", figsize=(5.3, 6.72), squeeze=False,
                                    gridspec_kw=gridspec_kw)

    # Iterate over the inputs x for evaluation
    for i_x, x_value in enumerate(x_values):

        # Plot bins
        colors_bin = colors[::-1].copy()
        colors_bin[:, -1] = alpha_bins

        for i_a, a in enumerate(range(0, dim_out)):
            axs_tau[i_x, 0].set_ylim([-0.1, 1.1])
            ec = "white" if i_a != centre_ind else dist_col
            zorder = -2 if i_a != centre_ind else -1
            axs_tau[i_x, 0].fill_between(x=[a - 0.5, a + 0.5], y1=1.0, fc=colors_bin[i_a], ec=ec, lw=2, zorder=zorder)

            if i_x == 0:
                string = r"$\frac{" + str(taus_all_frac[::-1][i_a].numerator) + "}{" + str(
                    taus_all_frac[::-1][i_a].denominator) + "}$"
                axs_tau[i_x, 0].text(a + 0.1, 0.8, string, ha="left", va="top", fontsize=12, rotation=0)

        # Get NN prediction
        x_test_raw = x_value * np.ones((n_taus, 1))
        x_test = np.concatenate([tau_mapping(np.expand_dims(taus_all, -1)), x_mapping(x_test_raw)], 1)
        k_draws = k_values[i_x]
        pred_test = model.predict(x_test)
        pred_test_ext = np.concatenate([np.zeros((n_taus, 1)), pred_test, np.ones((n_taus, 1))], axis=1)

        # Plot prediction
        for i_tau in range(n_taus):
            axs_tau[i_x, 0].step(bin_centres_ext, pred_test_ext[i_tau, :].cumsum(), alpha=alpha_pred, color=colors[i_tau], where="post")  # CDF

        # Get analytic CDF
        quantile_points = return_quantile_points(dim_out, k_draws)
        quantiles_analyt = return_quantiles(quantile_points, taus_all, dim_out, k_draws)

        # Print max. error
        print("x:", x_value)
        print("  max. error over all quantiles and bins:", np.abs(pred_test.cumsum(1) - quantiles_analyt).max())

        # Plot analytic median and quantile range if at least one expected count per bin
        if k_draws >= dim_out:
            median_analyt = return_quantiles(quantile_points, [0.5], dim_out, k_draws)[0]
            quantiles_high = quantiles_analyt[-1, :]
            quantiles_low = quantiles_analyt[0, :]
            y_errors = np.vstack([median_analyt - quantiles_low, quantiles_high - median_analyt])
            axs_tau[i_x, 0].step(bin_centres, median_analyt, where="post", color=median_col, ls=":", lw=1.5)
            eb = axs_tau[i_x, 0].errorbar(np.asarray(bin_centres)[:-1] + 0.1, y=median_analyt[:-1], fmt="none",
                                     yerr=y_errors[:, :-1], ls="none", capsize=12, capthick=pad_thick, ecolor=true_col)
            eb[2][0].set_lw(0)  # delete vertical line

            # Plot arrows
            for i in range(dim_out - 1):
                axs_tau[i_x, 0].arrow(x=bin_centres[i] + 0.1, y=quantiles_low[i]-arr_length, dx=0, dy=arr_length,
                                      color=true_col, length_includes_head=True, width=arr_width, lw=arr_lw)
                axs_tau[i_x, 0].arrow(x=bin_centres[i] + 0.1, y=quantiles_high[i]+arr_length, dx=0, dy=-arr_length,
                                      color=true_col, length_includes_head=True, width=arr_width, lw=arr_lw)

            axs_tau[i_x, 0].set_ylim([-0.13, 1.13])

            # if show_dist: plot flipped in second axis
            if show_dist:
                arr_length_1 = arr_length
                eb_1 = axs_tau[i_x, 1].errorbar(median_analyt[centre_ind:centre_ind+1], y=[arr_pos_flip], fmt="none",
                                              xerr=y_errors[:, centre_ind:centre_ind+1], ls="none", capsize=6, capthick=pad_thick,
                                              ecolor=true_col)
                eb_1[2][0].set_lw(0)  # delete vertical line
                axs_tau[i_x, 1].arrow(x=quantiles_low[centre_ind:centre_ind+1] - arr_length_1,
                                      y=arr_pos_flip, dx=arr_length_1, dy=0, color=true_col, length_includes_head=True, width=arr_width, lw=arr_lw)
                axs_tau[i_x, 1].arrow(x=quantiles_high[centre_ind:centre_ind + 1] + arr_length_1,
                                      y=arr_pos_flip, dx=-arr_length_1, dy=0, color=true_col, length_includes_head=True, width=arr_width, lw=arr_lw)

        # show_dist: also plot arrows for single draw
        if show_dist and k_draws == 1:
            arr_length_1 = arr_length
            eb_1 = axs_tau[i_x, 1].errorbar([0.5], y=[arr_pos_flip], fmt="none",
                                            xerr=[0.5], ls="none", capsize=6, capthick=pad_thick, ecolor=true_col)
            eb_1[2][0].set_lw(0)  # delete vertical line
            axs_tau[i_x, 1].arrow(x=0.0 - arr_length_1, y=arr_pos_flip, dx=arr_length_1, dy=0, color=true_col,
                                  length_includes_head=True, width=arr_width, lw=arr_lw)
            axs_tau[i_x, 1].arrow(x=1.0 + arr_length_1, y=arr_pos_flip, dx=-arr_length_1, dy=0, color=true_col,
                                  length_includes_head=True, width=arr_width, lw=arr_lw)

        # Plot the distribution of the cum. histogram in the central bin
        if show_dist:
            ax_CDF = axs_tau[i_x, 1]

            # For a single draw: plot circles to show discontinuity
            if k_draws == 1:
                x_vec = [-1, 0, 1, 2]

                # Plot CDF
                for i_x_ev, x_ev in enumerate(x_vec[:-1]):
                    ax_CDF.plot(x_vec[i_x_ev:i_x_ev + 2], 2 * [stats.binom.cdf(x_ev, k_draws, p_binom)], color="k")

                    if i_x_ev < len(x_vec) - 2:
                        ax_CDF.plot(x_vec[i_x_ev + 1], stats.binom.cdf(x_ev, k_draws, p_binom), ls="none", marker="o",
                                    mec="k", mfc="none", ms=6)
                        ax_CDF.plot(x_vec[i_x_ev + 1], stats.binom.cdf(x_vec[i_x_ev + 1], k_draws, p_binom),
                                    ls="none", marker="o", mec="k", mfc="k", ms=6)

                # Get x and y values
                eval_vec = taus_all  # y-values of CDF plot
                hit_vec = (eval_vec > (1 - p_binom)).astype(np.float32)  # x-values -> quantiles

            # For multiple draws
            else:
                # Plot CDF
                x_vec = np.hstack([x_lim[0], np.linspace(0, k_draws, k_draws + 1) / k_draws, x_lim[1]])
                ax_CDF.step(x_vec, stats.binom.cdf(k_draws * x_vec, k_draws, p_binom), color="k", where="post")

                # Get x and y values
                eval_vec = taus_all  # y-values of CDF plot
                hit_vec_ind = np.asarray(
                    [np.argmin(stats.binom.cdf(k_draws * x_vec, k_draws, p_binom) <= eval) for eval in eval_vec])  # indices where CDF and p-values cross
                hit_vec = x_vec[hit_vec_ind]  # x-values -> quantiles

            # Plot evaluation (analytic and NN prediction)
            for i_eval, eval in enumerate(eval_vec):
                ax_CDF.plot([x_lim[0], hit_vec[i_eval]], 2 * [eval], ls="--", color=colors[i_eval], alpha=0.5)
                ax_CDF.plot(2 * [hit_vec[i_eval]], [0, eval_vec[i_eval]], ls="--", color=colors[i_eval], alpha=0.5)
                ax_CDF.plot(2 * [pred_test_ext.cumsum(1)[i_eval, 1 + centre_ind]], [0, eval_vec[i_eval]],
                            color=colors[i_eval], alpha=alpha_pred)

    # Additional labels
    axs_tau[-1, 0].set_xticks(range(dim_out))
    axs_tau[-1, 0].set_xlabel("Bins")
    axs_tau[-2, 0].set_ylabel("Cumulative histogram")
    # axs_tau[0, 0].text(4.52, 0.8, r"$\tau$", ha="left", va="top", fontsize=16)
    axs_tau[0, 0].text(0.4, 0.35, r"$\tau$", ha="right", va="center", fontsize=16)

    for i_x in range(len(x_values)):
        axs_tau[i_x, 0].set_xlim([-0.75, dim_out - 0.25])
        axs_tau[i_x, 0].set_xticklabels(axs_tau[0, 0].get_xticks() + 1)
        axs_tau[i_x, 0].yaxis.set_major_locator(MultipleLocator(1))
        axs_tau[i_x, 0].yaxis.set_minor_locator(MultipleLocator(0.2))
        axs_tau[i_x, 0].tick_params(which="major", direction='out', length=8)
        axs_tau[i_x, 0].tick_params(which="minor", direction='out', length=4)

    if show_dist:
        for i_x in range(len(x_values)):
            ax_CDF = axs_tau[i_x, 1]
            ax_CDF.set_xlim(x_lim)
            ax_CDF.set_ylim(y_lim)
            ax_CDF.xaxis.set_major_locator(MultipleLocator(1))
            ax_CDF.xaxis.set_minor_locator(MultipleLocator(0.2))
            ax_CDF.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax_CDF.yaxis.set_major_locator(MultipleLocator(1))
            ax_CDF.yaxis.set_minor_locator(MultipleLocator(0.2))
            ax_CDF.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax_CDF.tick_params(which="major", direction='out', length=8)
            ax_CDF.tick_params(which="minor", direction='out', length=4)
            ax_CDF.yaxis.tick_right()
            # for child in ax_CDF.get_children():
            #     if isinstance(child, mpl.spines.Spine):
            #         child.set_color(dist_col)

        axs_tau[-1, 1].set_xlabel(r"$M_3$", fontdict={"family": "serif"}, labelpad=-4)
        axs_tau[-2, 1].set_ylabel(r"$F(M_3)$", labelpad=0)
        axs_tau[-2, 1].yaxis.set_label_position("right")

    plt.tight_layout(w_pad=0.25, h_pad=0)

    if show_k:
        for i_x in range(len(x_values)):
            r_circ = 0.36 * ((1.0 + x_values[i_x]) / 2.0)
            this_k = k_mapping(x_values_frac[i_x])
            # if i_x == 0:
            # axs_tau[0, 0].text(-0.275, 0.5, r"$k$", fontsize=14, alpha=1, ha="center", va="center", rotation=0)
            asp = get_aspect(axs_tau[i_x, 0])
            x_circ, y_circ = -0.275, 0.925
            circle = mpl.patches.Ellipse((x_circ, y_circ), r_circ * asp, r_circ, facecolor=true_col, edgecolor="white")
            axs_tau[i_x, 0].add_patch(circle)
            axs_tau[i_x, 0].text(x_circ, y_circ - 0.005, r"$" + str(this_k) + "$", fontsize=14, alpha=1, ha="center", va="center", rotation=0,
                                 color="white", size=9)
    fig_tau.savefig(os.path.join(model_path, "toy_example.pdf"), bbox_inches="tight")
