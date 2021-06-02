"""
Script for the bimodal example.
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
from matplotlib import pyplot as plt
import seaborn as sns
import colorcet as cc
from scipy.stats import norm
from scipy.stats import gaussian_kde
sns.set_style("ticks")
sns.set_context("talk")
tfd = tfp.distributions

# Set random seed for reproducibility
tf.random.set_seed(0)
np.random.seed(0)

# Set parameters
dtype_np = np.float32
batch_size = 2048  # batch size (2048)
dim_in = 3  # input dimension
dim_out = 10  # output dimension: no. of histogram bins
n_hidden = 256  # hidden neurons (256)
n_layers = 2  # number of hidden layers (2)
act_fun = "relu"  # activation function for the hidden layers ("relu")
act_fun_final = "softmax"  # activation function for the output layers ("softmax" for EMPL, "linear" for binwise_max_llh_cum)
do_batch_norm = False  # batch normalisation (False)
n_epochs = 10000  # number of epochs (= batch iterations in this case) for training (10000)
loss = 'EMPL'  # loss function: mean_absolute_error (bin-wise), x-ent, EM1, EMPL, or binwise_max_llh_cum
smoothing = 0.0  # smoothing for EMPL
tau_distribution = "uniform"  # distribution to draw tau's from during training
save_root = "./Models"
save_path = os.path.join(save_root, loss)
os.makedirs(save_root, exist_ok=True)

if loss == 'binwise_max_llh_cum':
    assert act_fun_final == "linear"

# Define mapping for inputs (0, 1) -> parameter range
alpha_mapping = lambda x: x  # mixture coefficient: determines probability for Gaussian 1 (P(Gaussian 2) = 1 - alpha)
noise_mapping = lambda x: 1.0 * x  # std of Gaussian noise: [0, 1] (note: BEFORE normalisation!)
tau_mapping = lambda t: (t - 0.5) * 12  # mapping for quantile level tau: [0, 1] -> [-6, 6]
x_mapping = lambda x: (x - 0.5) * 12  # mapping for input parameters seen by the NN: [0, 1] -> [-6, 6]


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
        # Generate input parameters
        x_raw = tf.random.uniform([self.batch_size, self.dim_in], 0.0, 1.0)
        y = self.get_label(x_raw)

        # Map x
        x = x_mapping(x_raw)

        # Return
        return x, y

    # k is determined as a function of the input
    @classmethod
    def get_label(cls, x):
        n_batch_x = x.shape[0]
        alphas, noise_levels = alpha_mapping(x[:, :1]), noise_mapping(x[:, 1:])
        line_1 = tf.ones([n_batch_x, dim_out]) * tf.range(dim_out, dtype=tf.float32) \
                 + tf.random.normal([n_batch_x, dim_out], mean=0.0, stddev=noise_levels[:, :1])
        line_2 = - tf.ones([n_batch_x, dim_out]) * tf.range(dim_out, dtype=tf.float32) \
                 + tf.random.normal([n_batch_x, dim_out], mean=0.0, stddev=noise_levels[:, 1:2])
        # Get either line_1 (P(alpha)) or line_2 (P(1-alpha))
        y_raw = tf.where(alphas > tf.random.uniform([n_batch_x, 1], minval=0, maxval=1), line_1, line_2)
        # Shift each sample to get non-neg. values
        y_shifted = y_raw - tf.math.reduce_min(y_raw, axis=1, keepdims=True)
        # Normalise
        hist = y_shifted / tf.math.reduce_sum(y_shifted, axis=1, keepdims=True)

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

def binwise_max_llh_cum_loss(p, p_hat, scope="binwise_max_llh_loss", reduction="", name="", logvar_clip_min=-15):
    """Compute binwise (Gaussian) max. llh loss"""
    with tf.name_scope(scope):
        y_true = p
        y_pred_logits, y_logvar = p_hat[:, :dim_out], p_hat[:, dim_out:]
        y_pred = tf.math.softmax(y_pred_logits, axis=1)
        ecdf_p = tf.math.cumsum(y_true, axis=1)
        ecdf_p_hat = tf.math.cumsum(y_pred, axis=1)
        if logvar_clip_min is not None:
            y_logvar = tf.clip_by_value(y_logvar, logvar_clip_min, np.infty)
        err = ecdf_p_hat - ecdf_p
        precision = tf.exp(-y_logvar)
        term1 = err ** 2 * precision
        term2 = y_logvar
        final_loss = tf.reduce_mean(tf.reduce_sum(term1 + term2, 1) / 2)
        return final_loss

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
dim_out_final = 2 * dim_out if loss == "binwise_max_llh_cum" else dim_out

model = keras.Sequential()
model.add(layers.Dense(n_hidden, activation=act_fun, input_shape=(dim_in_concat,)))
if do_batch_norm:
    model.add(layers.BatchNormalization())
for i_layer in range(n_layers - 1):
    model.add(layers.Dense(n_hidden, activation=act_fun))
    if do_batch_norm:
        model.add(layers.BatchNormalization())
model.add(layers.Dense(dim_out_final, activation=act_fun_final))
model.build()
model.summary()

# Select loss
if loss == "mean_absolute_error":
    loss_tf = loss
elif loss == "x-ent":
    loss_tf = keras.losses.CategoricalCrossentropy(from_logits=False)
elif loss == "binwise_max_llh_cum":
    loss_tf = binwise_max_llh_cum_loss
elif loss == "EM1":
    loss_tf = emd_loss
elif loss == "EMPL":
    loss_tf = EMPL(smoothing=smoothing)
else:
    raise NotImplementedError

# Load / train NN
if os.path.exists(os.path.join(save_path, "checkpoint")):
    # Load and compile
    model.load_weights(os.path.join(save_path, "weights"))
    model.compile(loss=loss_tf, optimizer='adam', metrics=[])

else:
    # Compile NN
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=n_epochs,
                                                                 decay_rate=0.1, staircase=False)
    model.compile(loss=loss_tf, optimizer='adam', metrics=[])

    history = model.fit(x=generator, epochs=n_epochs, verbose=2)
    os.makedirs(save_path, exist_ok=True)
    model.save_weights(os.path.join(save_path, "weights"))

    # Show training progress
    fig_history, ax_history = plt.subplots(1, 1)
    ax_history.plot(history.history["loss"])
    ax_history.set_xlabel("Epoch")
    ax_history.set_ylabel("Loss")

# Define bins
bin_centres = np.arange(dim_out)
bin_width = np.unique(np.diff(bin_centres))[0]

# # # # # # # #

if loss == "EMPL":
    # 1) Plot for a FIXED tau, show random realisations together with NN predictions
    # Evaluate NN for a fixed noise level, for different alphas
    noise_level_test = 0.5
    n_eval_alpha = 11
    x_test_alphas = np.linspace(0, 1, n_eval_alpha, dtype=dtype_np)
    x_test = np.hstack([np.expand_dims(x_test_alphas, -1), noise_level_test * np.ones((n_eval_alpha, 2))]).astype(dtype_np)
    tau_eval = 0.5
    y_test = generator.get_label(x_test)

    # Concatenate tau to input
    tau = tau_eval * np.ones((n_eval_alpha, 1), dtype=dtype_np)
    x_test = np.concatenate([tau_mapping(tau), x_mapping(x_test)], 1)

    # Predict
    pred_test = model.predict(x_test)

    # Plot
    fig_1, axs_1 = plt.subplots(3, 4, sharex="all", sharey="all", figsize=(16, 12))
    for i, ax in enumerate(axs_1.flatten()):
        if i >= n_eval_alpha:
            ax.axis("off")
            ax.text(0.5, 0.125, "Loss: " + loss.replace("_", " ").capitalize())
            break
        ax.bar(bin_centres, y_test[i], fc="darkgreen", ec="k", lw=2, alpha=0.5, width=1.0, align="center")
        ax.bar(bin_centres, pred_test[i], fc="firebrick", ec="k", lw=2, alpha=0.5, width=1.0, align="center")
        this_x_test = x_test[i][1:] if loss == "EMPL" else x_test[i]
        ax.set_title(str(np.round(this_x_test[0], 2)))
        ax.set_xticks(range(dim_out))
        fig_1.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.show()


# EVALUATION for EMPL and Gaussian max. llh loss (applied to cumulative histograms)
if loss in ["EMPL", "binwise_max_llh_cum"]:

    # EVALUATION SETTINGS
    alpha_test = 0.5  # SET ALPHA (PROBABILITY FOR FIRST MODE) FOR THE ENTIRE EVALUATION!
    noise_level_test = 0.5  # SET THE NOISE LEVEL FOR THE ENTIRE EVALUATION!
    true_samples = 65536

    # 2) Make a plot as a function of tau
    tau_vec = np.linspace(0.05, 0.95, 10)
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

    plot_pdf = True
    n_cols = 3 if plot_pdf else 2
    ax_inds_cdf = [1, 2] if plot_pdf else [0, 1]

    fig, axs = plt.subplots(1, n_cols, figsize=(6, 4), squeeze=False, sharex="row", sharey="none")
    i_row = 0
    width = 1.0
    bin_centres_plot = bin_centres + 1
    x_test_alphas = np.asarray([alpha_test, noise_level_test, noise_level_test])

    # Now: prediction and histogram plots
    if loss == "EMPL":
        # Tile for the different quantile levels
        x_test_tiled = np.tile(x_test_alphas[None], [n_taus] + [1])
        # Append taus
        x_test_mapped = np.concatenate([np.expand_dims(tau_mapping(tau_vec), -1), x_mapping(x_test_tiled)], 1).astype(dtype_np)
        # Predict
        pred_test = model.predict(x_test_mapped)
        pred_test_cum = np.cumsum(pred_test, 1)
    else:
        x_test_mapped = x_mapping(x_test_alphas[None]).astype(dtype_np)
        # Predict
        pred_test_raw = model.predict(x_test_mapped)
        pred_test_mean, pred_test_std = tf.math.softmax(pred_test_raw[:, :dim_out], 1).numpy(), np.exp(0.5 * pred_test_raw[:, dim_out:])
        pred_test_cum_mean = np.cumsum(pred_test_mean, 1)
        pred_test_cum = np.squeeze(np.asarray([pred_test_cum_mean + i_sigma * pred_test_std for i_sigma in n_sigmas]), 1)
        pred_test = np.tile(pred_test_mean, [n_taus, 1])  # llh estimation is directly for cum. histograms, no uncertainties for differential histograms available

    # Get samples
    samples = generator.get_label(np.tile(x_test_alphas[None], [true_samples, 1]).astype(dtype_np)).numpy()
    samples_cum = np.cumsum(samples, 1)
    true_quantiles = np.quantile(samples_cum, tau_vec, axis=0)

    # Iterate over the taus
    for i_tau in range(n_taus):

        if plot_pdf:
            # Plot differential histogram
            i_col = 0
            axs[i_row, i_col].fill_between(bin_centres_plot - width / 2.0, pred_test[i_tau, :], color=colors[i_tau],
                                            zorder=1, alpha=0.075, step="post")

            # Plot median
            if np.abs(tau_vec[i_tau] - 0.5) < 0.001:
                axs[i_row, i_col].step(bin_centres_plot - width / 2.0, pred_test[i_tau, :], color="k", lw=2, zorder=3,
                                        alpha=1.0, where="post")

        # Plot predicted cumulative histogram
        for i_col, cum_hist in zip(ax_inds_cdf, [pred_test_cum, true_quantiles]):

            if i_tau < n_taus - 1:
                # Draw the next section of the cumulative histogram in the right colour
                for i in range(len(bin_centres_plot)):
                    # Draw the next section of the cumulative histogram in the right colour
                    axs[i_row, i_col].fill_between(x=[bin_centres_plot[i] - width / 2.0, bin_centres_plot[i] + width / 2.0],
                                              y1=cum_hist[i_tau, i],
                                              y2=cum_hist[i_tau + 1, i], color=colors[i_tau], lw=0)
                    # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                    if i_tau == 0 and cum_hist[i_tau, i] > 0.99:
                        axs[i_row, i_col].plot([bin_centres_plot[i] - width / 2.0, bin_centres_plot[i] + width / 2.0], 2 * [1.0],
                                          color=colors[0], lw=2, zorder=3)
                    elif i_tau == n_taus - 2 and cum_hist[i_tau, i] < 0.01:
                        axs[i_row, i_col].plot([bin_centres_plot[i] - width / 2.0, bin_centres_plot[i] + width / 2.0], 2 * [0.0],
                                          color=colors[-1], lw=2, zorder=3)

    # Set axes limits
    axs[i_row, ax_inds_cdf[0]].set_ylim([-0.075, 1.075])
    axs[i_row, ax_inds_cdf[1]].set_ylim([-0.075, 1.075])
    if plot_pdf:
        axs[i_row, 0].set_ylim([-0.025, 0.35])
    plt.show()
    out_path_tmp = os.path.join(save_path, "histogram_data_" + str(alpha_test) + "_" + str(noise_level_test))
    np.savez(out_path_tmp, true_quantiles=true_quantiles, samples_cum=samples_cum,
             pred_test_cum=pred_test_cum, x_test_alphas=x_test_alphas, alpha_test=alpha_test, noise_level_test=noise_level_test,
             tau_vec=tau_vec, bin_centres_plot=bin_centres_plot)

    ## Calibration plot
    coverage_over_random_inputs = False  # compute coverage over random inputs or for fixed alpha_test and noise_level_test?
                                         # if True: samples_cal random inputs are drawn and coverage is computed based on them
                                         # if False: calibration is computed for NN predictions at fixed alpha_test &
                                         # noise_level_test, evaluated for true_samples randomly generated labels
                                         # NOTE: if True, opposite errors for different inputs X may cancel each other and
                                         # give a decent calibration plot despite terrible results for each fixed X!
    samples_cal = 65536
    cutoff = 1e-5  # if CDF is < cutoff or > 1 - cutoff: ignore in order not to bias results by irrelevant bins
    tol = 0.0
    beta_vec = np.linspace(0.025, 0.475, 19)
    n_betas = len(beta_vec)
    taus = np.hstack([0.5 - beta_vec[::-1], 0.5 + beta_vec])
    n_taus = len(taus)
    n_sigmas = norm.ppf(taus)
    beta_range_vec = (taus[::-1] - taus)[:n_betas]
    coverage = np.zeros(n_betas)

    if coverage_over_random_inputs:
        x_test_alphas = np.random.uniform(size=(samples_cal, dim_in), low=0.0, high=1.0)
    else:
        x_test_alphas = np.asarray([alpha_test, noise_level_test, noise_level_test])

    if loss == "EMPL":
        if coverage_over_random_inputs:
            # Tile for the different quantile levels
            x_test_tiled_3d = np.tile(x_test_alphas[None], [n_taus, 1, 1])
            # Append taus
            x_test_mapped_3d = np.concatenate([np.tile(tau_mapping(taus), [1, samples_cal, 1]).transpose([2, 1, 0]),
                                               x_mapping(x_test_tiled_3d)], -1).astype(dtype_np)
            # 3D -> 2D
            x_test_mapped = x_test_mapped_3d.reshape([-1, dim_in + 1])
        else:
            x_test_tiled = np.tile(x_test_alphas[None], [n_taus] + [1])
            # Append taus
            x_test_mapped = np.concatenate([np.expand_dims(tau_mapping(taus), -1), x_mapping(x_test_tiled)], 1).astype(dtype_np)

        # Predict
        pred_test = model.predict(x_test_mapped)
        pred_test_cum = np.cumsum(pred_test, 1)
    else:
        if coverage_over_random_inputs:
            x_test_mapped = x_mapping(x_test_alphas).astype(dtype_np)
        else:
            x_test_mapped = x_mapping(x_test_alphas[None]).astype(dtype_np)

        # Predict
        pred_test_raw = model.predict(x_test_mapped)
        pred_test_mean, pred_test_std = tf.math.softmax(pred_test_raw[:, :dim_out], 1).numpy(), np.exp(0.5 * pred_test_raw[:, dim_out:])
        pred_test_cum_mean = np.cumsum(pred_test_mean, 1)
        if coverage_over_random_inputs:
            pred_test_cum = np.asarray([pred_test_cum_mean + i_sigma * pred_test_std for i_sigma in n_sigmas])
        else:
            pred_test_cum = np.squeeze(np.asarray([pred_test_cum_mean + i_sigma * pred_test_std for i_sigma in n_sigmas]), 1)
        pred_test = np.tile(pred_test_mean, [n_taus, 1])  # llh estimation is directly for cum. histograms, no uncertainties for differential histograms available

    # Get samples
    if coverage_over_random_inputs:
        samples = generator.get_label(x_test_alphas.astype(dtype_np)).numpy()
        samples_cum = np.cumsum(samples, 1)
    else:
        samples = generator.get_label(np.tile(x_test_alphas[None], [true_samples, 1]).astype(dtype_np)).numpy()
        samples_cum = np.cumsum(samples, 1)

    # If random samples: need to reshape
    if coverage_over_random_inputs:
        pred_test_cum_eval = np.reshape(pred_test_cum, [n_taus, samples_cal, dim_out])
    else:
        pred_test_cum_eval = np.expand_dims(pred_test_cum, 1)

    # Get coverage
    for i_beta in range(len(beta_vec)):
        within_q_range = np.logical_and(samples_cum >= pred_test_cum_eval[i_beta, :, :] - tol,
                                        samples_cum <= pred_test_cum_eval[n_taus - 1 - i_beta, :, :] + tol)
        outside_inds = list(np.nonzero(np.logical_or(samples_cum < cutoff, samples_cum > 1 - cutoff)))
        within_q_range = within_q_range.astype(np.float32)
        within_q_range[outside_inds[0], outside_inds[1]] = np.nan
        coverage[i_beta] = np.nanmean(within_q_range)
    print(coverage)

    # Make plot
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", lw=1, color="darkblue")
    markers = ["o"]
    mfcs = ["k"]
    mss = [8]
    mews = [1]
    alphas_plot = [1]
    for i_method in range(1):
        ax.plot(beta_range_vec, coverage, "k", marker=markers[i_method], lw=0, markersize=mss[i_method], mfc=mfcs[i_method],
                markeredgewidth=mews[i_method], label=loss, alpha=alphas_plot[i_method])
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(r"Confidence level $\alpha$")
    ax.set_ylabel(r"Coverage $p_{\mathrm{cov}}(\alpha)$")
    ax.set_aspect("equal")
    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.legend()
    plt.tight_layout()
    plt.show()
    out_path_tmp = os.path.join(save_path, "calibration_" + str(alpha_test) + "_" + str(noise_level_test))
    np.savez(out_path_tmp, beta_range_vec=beta_range_vec, coverage=coverage, tol=tol,
             samples_cum=samples_cum, pred_test_cum_eval=pred_test_cum_eval, beta_vec=beta_vec, cutoff=cutoff,
             x_test_alphas=x_test_alphas, coverage_over_random_inputs=coverage_over_random_inputs, samples_cal=samples_cal)

    ## Plot distribution in a selected bin (note: 0-based indexing)
    bin_plot = 4
    taus_bin = np.linspace(0, 1, 201)
    n_sigmas_bin = norm.ppf(taus_bin)
    n_taus_bin = len(taus_bin)
    x_test_alphas = np.asarray([alpha_test, noise_level_test, noise_level_test])

    if loss == "EMPL":
        # Tile for the different quantile levels
        x_test_tiled = np.tile(x_test_alphas[None], [n_taus_bin] + [1])
        # Append taus
        x_test_mapped = np.concatenate([np.expand_dims(tau_mapping(taus_bin), -1), x_mapping(x_test_tiled)], 1).astype(dtype_np)
        # Predict
        pred_test = model.predict(x_test_mapped)
        pred_test_cum = np.cumsum(pred_test, 1)
    else:
        x_test_mapped = x_mapping(x_test_alphas[None]).astype(dtype_np)
        # Predict
        pred_test_raw = model.predict(x_test_mapped)
        pred_test_mean, pred_test_std = tf.math.softmax(pred_test_raw[:, :dim_out], 1).numpy(), np.exp(0.5 * pred_test_raw[:, dim_out:])
        pred_test_cum_mean = np.cumsum(pred_test_mean, 1)
        pred_test_cum = np.squeeze(np.asarray([pred_test_cum_mean + i_sigma * pred_test_std for i_sigma in n_sigmas_bin]), 1)
        pred_test = np.tile(pred_test_mean, [n_taus, 1])  # llh estimation is directly for cum. histograms, no uncertainties for differential histograms available

    # Make a plot
    plot_points = np.linspace(0, 1, 201)
    kde = gaussian_kde(samples_cum[:, bin_plot], bw_method=0.015)
    fig_bin_cum, ax_bin_cum = plt.subplots(1, 1, figsize=(5, 5))
    ax_bin_cum.plot(pred_test_cum[:, bin_plot], taus_bin, lw=0, marker="o", markersize=6)
    kde_int = np.asarray([kde.integrate_box_1d(0, p) for p in plot_points])
    ax_bin_cum.plot(plot_points, kde_int, lw=2, color="k")
    plt.show()
    out_path_tmp = os.path.join(save_path, "bin_data_" + str(alpha_test) + "_" + str(noise_level_test))
    np.savez(out_path_tmp, plot_points=plot_points, pred_test_cum=pred_test_cum, kde_int=kde_int,
             bin_plot=bin_plot, taus_bin=taus_bin)
