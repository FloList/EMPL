"""
This script is the main script for the Bundesliga example.
Once the data is generated and bootstrapping uncertainties have been calculating using the scripts
* make_bundeliga_table.py and
* make_leave_one_out_hists.py,
run this file to train / load the NN.
"""
import os
import time
from shutil import copyfile
import keras
from keras import layers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
from copy import copy
import numpy as np
import matplotlib as mpl
from matplotlib import colors
import colorcet as cc
mpl.use('Qt5Agg')
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")
# # # # # # # # # #
# Set random seeds for reproducibility
tf.random.set_seed(0)
np.random.seed(0)

# Input as histogram or as array?
HIST_MODE = False  # TURN OFF! Using a histogram for the input leads to the order of the matches being lost!

# Load data
data_file_train = "train_data.npz"
data_file_val = "val_data.npz"
data_file_loo = "leave_one_out_data_all.npy"
y_hist_file_train = "y_hist_train.npy"
y_hist_file_val = "y_hist_val.npy"
train_data = np.load(data_file_train, allow_pickle=True)
val_data = np.load(data_file_val, allow_pickle=True)
n_augment, n_seasons, n_matches, n_clubs = train_data["table"].shape
n_seasons_val = val_data["table"].shape[0]
do_initial_plot = True  # plot some examples before training/loading NN
do_median_plot = True  # after training/loading NN: make a plot of the median prediction
do_input_histogram = False  # if True: plot input as a histogram, rather than for each week

# Set parameters
batch_size = 2048  # batch size
dim_out = n_clubs  # output dimension: no. of histogram bins (number of teams in league)
n_hidden = 128  # hidden neurons
n_layers = 2  # number of hidden layers
act_fun = "relu"  # activation function for the hidden layers
act_fun_final = "softmax"  # activation function for the output layers
do_batch_norm = False  # batch normalisation
dropout_rate = 0.5  # dropout rate
n_epochs = 250  # number of epochs for training
loss = 'EMPL'  # loss function: x-ent / mean_absolute_error / mean_squared_error (bin-wise), EM1 / EM2, or EMPL / EMPL_2, binwise_max_llh_cum
smoothing = 0.005  # smoothing for EMPL
tau_distribution = "uniform"  # distribution to draw tau's from during training
save_path = "./Models"
os.makedirs(save_path, exist_ok=True)

if loss == 'binwise_max_llh':
    assert act_fun_final == "linear"

# Set parameters depending on loss choices
r_EM = 2
if loss == 'EM1':
    r_EM = 1
r_pinball = 1
if loss == "EMPL_2":
    r_pinball = 2

# Prepare data
input_field = "points"  # CHOOSE INPUT HERE!
x_raw_train = train_data[input_field]
y_raw_train = train_data["table"]
x_raw_val = val_data[input_field]
y_raw_val = val_data["table"]

x_pre_train = np.reshape(np.transpose(x_raw_train, [0, 1, 3, 2]), [-1, n_matches])  # n_augment * n_seasons * n_clubs  x  n_matches
y_pre_train = np.reshape(np.transpose(y_raw_train, [0, 1, 3, 2]), [-1, n_matches])  # n_augment * n_seasons * n_clubs  x  n_matches
x_pre_val = np.reshape(np.transpose(x_raw_val, [0, 2, 1]), [-1, n_matches])  # n_seasons * n_clubs  x  n_matches
y_pre_val = np.reshape(np.transpose(y_raw_val, [0, 2, 1]), [-1, n_matches])  # n_seasons * n_clubs  x  n_matches

# Inputs
if HIST_MODE:
    bin_lims = (min(x_raw_train.min(), x_raw_val.min()), max(x_raw_train.max(), x_raw_val.max()))
    hist_bins_x = np.arange(bin_lims[0], bin_lims[1] + 2)  # histogram input for scored goals
    x_hist_train = np.asarray([np.histogram(x_pre_train[i, :], bins=hist_bins_x, density=True)[0] \
                         for i in range(n_augment * n_seasons * n_clubs)])  # histogram for scored goals
    x_hist_val = np.asarray([np.histogram(x_pre_val[i, :], bins=hist_bins_x, density=True)[0] \
                         for i in range(n_seasons_val * n_clubs)])
else:
    # Preprocessing for goals / points
    approx_mean_points = 1.4
    x_no_hist_train = x_pre_train - approx_mean_points
    x_no_hist_val = x_pre_val - approx_mean_points

# Labels
hist_bins = np.arange(1, n_clubs + 2)
if os.path.exists(y_hist_file_train):
    y_hist_train = np.load(y_hist_file_train)
else:
    y_hist_train = np.asarray([np.histogram(y_pre_train[i, :], bins=hist_bins, density=True)[0] \
                               for i in range(n_augment * n_seasons * n_clubs)])  # n_augment * n_seasons * n_clubs  x  n_clubs (standings)
    np.save(y_hist_file_train, y_hist_train)
if os.path.exists(y_hist_file_val):
    y_hist_val = np.load(y_hist_file_val)
else:
    y_hist_val = np.asarray([np.histogram(y_pre_val[i, :], bins=hist_bins, density=True)[0] \
                             for i in range(n_seasons_val * n_clubs)])  # n_seasons * n_clubs  x  n_clubs (standings)
    np.save(y_hist_file_val, y_hist_val)

# Get club names
club_names_train = np.reshape(train_data["club_names"], [-1])
club_names_val = np.reshape(val_data["club_names"], [-1])

# Get seasons
years_val = [1998, 2006, 2014]
years_train = np.setdiff1d(np.arange(1995, 2018), years_val)
seasons_train = np.reshape(np.tile(np.expand_dims([str(y) + "-" + str(int(y+1))[-2:].zfill(2) for y in years_train], 1), [1, n_clubs]), [-1])
seasons_val = np.reshape(np.tile(np.expand_dims([str(y) + "-" + str(int(y+1))[-2:].zfill(2) for y in years_val], 1), [1, n_clubs]), [-1])

final_pos_train = y_raw_train[:, :, -1, :]
final_pos_val = y_raw_val[:, -1, :]
final_pos_val_resh = np.reshape(final_pos_val, [-1])

goal_bins = np.arange(-10, 10)

# Plot some data:
if do_initial_plot:
    n_plot = 5
    fig_data, axs_data = plt.subplots(n_plot, 3, figsize=(14, 14), sharex="col", sharey="col")

    for i_plot in range(n_plot):
        if HIST_MODE:
            axs_data[i_plot, 0].bar(hist_bins_x[:-1], x_hist_val[i_plot], color="darkblue")
        else:
            axs_data[i_plot, 0].hist(x_pre_val[i_plot], bins=goal_bins, color="darkblue")
        axs_data[i_plot, 1].bar(hist_bins[:-1], y_hist_val[i_plot], color="darkgreen")
        axs_data[i_plot, 2].bar(hist_bins[:-1], y_hist_val[i_plot].cumsum(), color="darkgreen")

    for i_plot in range(n_plot):
        axs_data[i_plot, 0].text(axs_data[i_plot, 0].get_xlim()[0] + 0.2, 0.9 * axs_data[i_plot, 0].get_ylim()[1],
                                 seasons_val[i_plot] + ":\n   " + club_names_val[i_plot], va="top", ha="left")
        axs_data[i_plot, 1].text(n_clubs - 0.2, 0.9 * axs_data[i_plot, 1].get_ylim()[1], str(int(final_pos_val_resh[i_plot])), va="top", ha="right")
        axs_data[i_plot, 1].axvline(final_pos_val_resh[i_plot], color="0.5", ls="--")
    plt.subplots_adjust(wspace=0.2, hspace=0.0)

# Define tau mapping for input
tau_mapping = lambda t: (t - 0.5) * 12

# Helper function to append the quantile level tau
def append_tau(x, y, tau, do_tf=True):
    if do_tf:
        x = tf.concat([tau_mapping(tau), x], axis=1)
        if y is not None:
            y = tf.concat([tau, y], axis=1)
    else:
        x = np.concatenate([tau_mapping(tau), x], axis=1)
        if y is not None:
            y = np.concatenate([tau, y], axis=1)
    return x, y

# Helper function to remove the quantile level tau
def remove_tau(x, y):
    x = x[:, 1:]
    if y is not None:
        y = y[:, 1:]
    return x, y


# Loss functions
def emd_loss(p, p_hat, scope="emd_loss", do_root=True):
    """Compute the Earth Mover's Distance loss."""
    with tf.name_scope(scope):
        ecdf_p = tf.math.cumsum(p, axis=-1)
        ecdf_p_hat = tf.math.cumsum(p_hat, axis=-1)
        if r_EM == 1:
            emd = tf.reduce_mean(tf.abs(ecdf_p - ecdf_p_hat), axis=-1)
        elif r_EM == 2:
            emd = tf.reduce_mean((ecdf_p - ecdf_p_hat) ** 2, axis=-1)
            if do_root:
                emd = tf.sqrt(emd)
        else:
            emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), r_EM), axis=-1)
            if do_root:
                emd = tf.pow(emd, 1 / r_EM)
        return tf.reduce_mean(emd)


def EMPL(p, p_hat, scope="empl", reduction="", name=""):
    """Compute the EMPL"""
    with tf.name_scope(scope):
        tau, y_true = p[:, :1], p[:, 1:]
        y_pred = p_hat
        ecdf_p = tf.math.cumsum(y_true, axis=-1)
        ecdf_p_hat = tf.math.cumsum(y_pred, axis=-1)
        delta = ecdf_p_hat - ecdf_p

        if r_pinball == 1:
            # Non-smooth C0 loss (default)
            if smoothing == 0.0:
                mask = tf.cast(tf.greater_equal(delta, tf.zeros_like(delta)), tf.float32) - tau
                loss = mask * delta

            # Smooth loss
            else:
                loss = -tau * delta + smoothing * tf.math.softplus(delta / smoothing)

        elif r_pinball == 2:
            mask = tf.cast(tf.greater_equal(delta, tf.zeros_like(delta)), tf.float32) - tau
            loss = tf.abs(mask) * delta ** 2.0

        else:
            raise NotImplementedError

        mean_loss = tf.reduce_mean(loss, axis=-1)
        final_loss = tf.reduce_mean(mean_loss)

        return tf.reduce_mean(final_loss)


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
dim_in = n_matches if not HIST_MODE else len(hist_bins_x) - 1
dim_in_concat = dim_in + 1 if "EMPL" in loss else dim_in

model = keras.Sequential()
model.add(layers.Dense(n_hidden, activation=act_fun, input_shape=(dim_in_concat,)))
if dropout_rate > 0.0:
    model.add(layers.Dropout(rate=dropout_rate))
if do_batch_norm:
    model.add(layers.BatchNormalization())

for i_layer in range(n_layers - 1):
    model.add(layers.Dense(n_hidden, activation=act_fun))
    if dropout_rate > 0.0:
        model.add(layers.Dropout(rate=dropout_rate))
    if do_batch_norm:
        model.add(layers.BatchNormalization())

dim_out_final = 2 * dim_out if loss == "binwise_max_llh_cum" else dim_out
model.add(layers.Dense(dim_out_final, activation=act_fun_final))
model.build()
model.summary()

# Select loss
if loss in ["mean_absolute_error", "mean_squared_error"]:
    loss_tf = loss
elif loss in ["EM1", "EM2"]:
    loss_tf = emd_loss
elif loss in ["EMPL", "EMPL_2"]:
    loss_tf = EMPL
elif loss == "x-ent":
    loss_tf = keras.losses.CategoricalCrossentropy(from_logits=False)
elif loss == "binwise_max_llh_cum":
    loss_tf = binwise_max_llh_cum_loss
else:
    raise NotImplementedError

# Compile NN
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=n_epochs, decay_rate=0.1, staircase=False)

# For EMPL: append taus
if "EMPL" in loss:
    tau_train = tf.random.uniform((n_seasons * n_augment * n_clubs, 1), 0.0, 1.0)
    tau_val = 0.5 * np.ones((n_seasons_val * n_clubs, 1))
    if HIST_MODE:
        x_train, y_train = append_tau(x_hist_train, y_hist_train, tau_train, do_tf=True)
        x_val, y_val = append_tau(copy(x_hist_val), copy(y_hist_val), tau_val, do_tf=False)
    else:
        x_train, y_train = append_tau(x_no_hist_train, y_hist_train, tau_train, do_tf=True)
        x_val, y_val = append_tau(copy(x_no_hist_val), copy(y_hist_val), tau_val, do_tf=False)
else:
    if HIST_MODE:
        x_train, y_train = x_hist_train, y_hist_train
        x_val, y_val = copy(x_hist_val), copy(y_hist_val)
    else:
        x_train, y_train = x_no_hist_train, y_hist_train
        x_val, y_val = copy(x_no_hist_val), copy(y_hist_val)


# Train / load NN
model_path = os.path.join(save_path, loss)
if loss == "EMPL" and smoothing > 0:
    model_path += "_" + str(smoothing)
if os.path.exists(model_path):
    # Load and compile
    model.load_weights(os.path.join(model_path, "weights"))
    model.compile(loss=loss_tf, optimizer='adam', metrics=[])
else:
    # Compile
    model.compile(loss=loss_tf, optimizer='adam', metrics=[])

    # Back up this file
    datetime = time.ctime().replace("  ", "_").replace(" ", "_").replace(":", "-")
    os.makedirs(model_path, exist_ok=True)
    file_backup = os.path.join(model_path, "train_file_" + datetime + ".py")
    copyfile("Bundesliga_example.py", file_backup)

    # Train and save
    history = model.fit(x=x_train, y=y_train, epochs=n_epochs, verbose=2, batch_size=batch_size, validation_data=(x_val, y_val))
    model.save_weights(os.path.join(model_path, "weights"))

    # Show training progress
    fig_history, ax_history = plt.subplots(1, 1)
    ax_history.plot(history.history["loss"])
    ax_history.plot(history.history["val_loss"])
    ax_history.set_xlabel("Epoch")
    ax_history.set_ylabel("Loss")
    ax_history.legend(["Training", "Validation"])
    fig_history.savefig(os.path.join(model_path, "training.pdf"), bbox_inches="tight")
    plt.close(fig_history)

# Predict median for all validation samples
pred_median_all = model.predict(x_val)

# if EMPL: need to remove quantiles for plotting
if "EMPL" in loss:
    x_val, y_val = remove_tau(x_val, y_val)

# if binwise max. llh loss:
if loss == "binwise_max_llh_cum":
    pred_median_all, pred_logvar = tf.math.softmax(pred_median_all[:, :dim_out], 1).numpy(), pred_median_all[:, dim_out:]

# Stats for median
l1_error = np.abs(pred_median_all - y_val).mean()
l2_error = np.sqrt(((pred_median_all - y_val) ** 2).mean(1)).mean()

ecdf_p = np.cumsum(y_val, axis=-1)
ecdf_p_hat = np.cumsum(pred_median_all, axis=-1)
emd_1_error = np.mean(np.sum(np.abs(ecdf_p - ecdf_p_hat), axis=1))
emd_2_error = np.sqrt(np.sum((ecdf_p - ecdf_p_hat) ** 2, axis=1)).mean()


def ACCJS(p_, q_, eps=1e-8):
    # if p(i) = 0 then ACCJS(p, q)(i) = 0 since xlog(x) -> 0 as x-> 0
    p_ = np.clip(p_, eps, 1.0)
    return 0.5 * np.sum(p_ * np.log(p_ / (0.5 * (p_ + q_))), axis=-1)


cjs_error = np.mean(ACCJS(ecdf_p, ecdf_p_hat) + ACCJS(ecdf_p_hat, ecdf_p))

hist_intersec_error_all = np.nansum(np.stack([y_val, pred_median_all], 2).min(2), 1)
hist_intersec_error = hist_intersec_error_all.mean()
plt.hist(hist_intersec_error_all, bins=np.linspace(0, 1, 11))
plt.title("Distribution of histogram intersections")
plt.xlabel("Intersection")
plt.ylabel("Frequency")


with open(os.path.join(model_path, "stats_out_txt"), 'w') as f:
    print('Mean statistics:\n l1 error: ' + str(l1_error) + '\n l2 error: ' + str(l2_error) + '\n EMD 1 error: '
          + str(emd_1_error) + '\n EMD 2 error: ' + str(emd_2_error) + '\n CJS error ' + str(cjs_error) +
          '\n Intersection: ' + str(hist_intersec_error), file=f)

# Plot median prediction
if do_median_plot:
    n_show = 4
    fig_eval, axs_eval = plt.subplots(n_show, 3, figsize=(14, 14), sharex="col", sharey="col")
    for i_eval in range(n_show):
        if HIST_MODE:
            axs_eval[i_eval, 0].bar(hist_bins_x[:-1], x_val[i_eval], color="darkblue")
        else:
            axs_eval[i_eval, 0].hist(np.asarray(x_val[i_eval]), bins=len(goal_bins), fc="darkblue")
        axs_eval[i_eval, 1].bar(hist_bins[:-1], y_val[i_eval], color="darkgreen", alpha=0.5)
        axs_eval[i_eval, 1].step(hist_bins[:-1], pred_median_all[i_eval], color="firebrick", alpha=1, where="mid")
        axs_eval[i_eval, 2].bar(hist_bins[:-1], np.cumsum(y_val[i_eval]), color="darkgreen", alpha=0.5)
        axs_eval[i_eval, 2].step(hist_bins[:-1], np.cumsum(pred_median_all[i_eval]), color="firebrick", alpha=1, where="mid")
        if loss == "binwise_max_llh_cum":
            axs_eval[i_eval, 2].errorbar(x=hist_bins[:-1], y=np.cumsum(pred_median_all[i_eval]),
                                         yerr=np.exp(0.5 * pred_logvar[i_eval]), color="firebrick", alpha=1, capsize=5)
    plt.subplots_adjust()

    for i_eval in range(n_show):
        axs_eval[i_eval, 0].text(axs_eval[i_eval, 0].get_xlim()[0] + 0.2, 0.9 * axs_eval[i_eval, 0].get_ylim()[1],
                                 seasons_val[i_eval] + ":\n   " + club_names_val[i_eval], va="top", ha="left")
        axs_eval[i_eval, 1].text(n_clubs - 0.2, 0.9 * axs_eval[i_eval, 1].get_ylim()[1], str(int(final_pos_val_resh[i_eval])), va="top", ha="right")
        axs_eval[i_eval, 1].axvline(final_pos_val_resh[i_eval], color="0.5", ls="--")
    plt.subplots_adjust(wspace=0.2, hspace=0.0)
    fig_eval.savefig(os.path.join(model_path, "median_plot.pdf"), bbox_inches="tight")
    plt.close(fig_eval)


# Now, consider different quantiles
if "EMPL" in loss or loss == "binwise_max_llh_cum":
    shuffle = 2  # 0: don't shuffle, 1: shuffle randomly, 2: for paper
    if shuffle == 0:
        eval_inds = np.arange(x_val.shape[0])
    elif shuffle == 1:
        eval_inds = np.random.permutation(x_val.shape[0])
    elif shuffle == 2:
        eval_inds = [33, 3, 42]   #  33: VfB 2006-07 (1st), 3: Duisburg 1998-99 (8th), 42: HSV 2014-15 (16th)
    else:
        raise NotImplementedError

    # Load leave-one-out histograms
    show_leave_one_out = False
    if show_leave_one_out:
        hist_val_aug_all = np.load(data_file_loo)
    n_eval_q = 3
    all_preds_cum = []

    # Additional data?
    # add_data_path = './binwise_max_llh_cum/all_preds_cum.npy'
    # add_data = np.load(add_data_path)
    add_data = None

    # Define taus
    if "EMPL" in loss:
        taus = np.linspace(0.1, 0.9, 9)
    else:
        taus = [0.1, 0.5, 0.9]  # 80% prob. mass around mean of Gaussian
        n_sigmas = [-1.28155, 0, 1.28155]
    n_taus = len(taus)

    # Set colours
    colours = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]

    # Make plot
    sns.set_context("paper")
    fig_tau, axs_tau = plt.subplots(n_eval_q, 3, sharex="col", sharey="col", figsize=(14, 14))
    for i_eval_q, eval_ind in enumerate(eval_inds[:n_eval_q]):

        # Get x values and tile (for all quantile levels)
        x_val_q = x_val[eval_ind:eval_ind + 1] * np.ones((n_taus, 1))

        # EMPL
        if "EMPL" in loss:
            # Append quantile level, predict, and remove quantile level
            x_val_q, _ = append_tau(x_val_q, None, np.expand_dims(taus, 1))
            pred = model.predict(x_val_q)
            x_val_q, _ = remove_tau(x_val_q, None)
            pred_cum = pred.cumsum(1)

        # Binwise max. Gaussian llh for cumulative histogram
        else:
            pred_raw = model.predict(x_val[eval_ind:eval_ind+1])
            pred_mean, pred_std = tf.math.softmax(pred_raw[:, :dim_out], 1).numpy(), np.exp(0.5 * pred_raw[:, dim_out:])
            pred_cum_mean = pred_mean.cumsum(1)
            pred_cum = np.squeeze(np.asarray([pred_cum_mean + i_sigma * pred_std for i_sigma in n_sigmas]), 1)
            pred = np.tile(pred_mean, [n_taus, 1])   # llh estimation is directly for cum. histograms, no uncertainties for differential histograms available

        all_preds_cum.append(pred_cum)

        # Get leave-one-out (cumulative) histogram
        if show_leave_one_out:
            hist_val_aug = hist_val_aug_all[:, eval_ind, :]
            hist_val_aug_cum = hist_val_aug.cumsum(1)

        x_val_q_0 = x_val_q[0] if isinstance(x_val_q, np.ndarray) else x_val_q[0].numpy()

        # Plot histogram of the inputs
        if do_input_histogram or HIST_MODE:  # input is already a histogram -> bar
            axs_tau[i_eval_q, 0].bar(hist_bins_x[:-1], x_val_q_0, color="darkblue")
        elif do_input_histogram and not HIST_MODE:  # make a histogram
            axs_tau[i_eval_q, 0].hist(x_val_q_0 + approx_mean_points, bins=len(goal_bins), fc="darkblue")
        else:  # not a histogram, but imshow
            cmap = colors.ListedColormap(['firebrick', '0.6', "darkgreen"])
            bounds = [-0.5, 0.5, 1.5, 3.5]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            inputs = x_val_q_0[None] + approx_mean_points
            axs_tau[i_eval_q, 0].imshow(inputs, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
            axs_tau[i_eval_q, 0].set_xticks(np.arange(x_val_q.shape[1] + 1) - 0.5)
            axs_tau[i_eval_q, 0].set_yticks([-0.5, 0.5])
            axs_tau[i_eval_q, 0].grid(which='major', color='w', linestyle='-', linewidth=1)
            axs_tau[i_eval_q, 0].set_ylim([-0.5, 0.5])
            axs_tau[i_eval_q, 0].set_xticklabels("")
            axs_tau[i_eval_q, 0].set_yticklabels("")
            axs_tau[i_eval_q, 0].tick_params(direction='out', length=0, width=0, colors='none', grid_color='white', grid_alpha=1.0)

        # Plot histogram and cumulative histogram
        axs_tau[i_eval_q, 1].bar(hist_bins[:-1], y_val[eval_ind], color="orange", alpha=0.5)
        cum_col_1 = [0.25490196, 0.71372549, 0.76862745, 1]
        cum_col_01 = [0.25490196, 0.71372549, 0.76862745, 0.1]
        axs_tau[i_eval_q, 2].bar(hist_bins[:-1], np.cumsum(y_val[eval_ind]), fc=cum_col_01, ec=cum_col_1, width=1, lw=2)

        # Plot predicted quantiles
        for i_tau, tau in enumerate(taus):
            # Plot histogram
            axs_tau[i_eval_q, 1].step(hist_bins[:-1], pred[i_tau], color=colours[i_tau], alpha=1, where="mid")
            # Plot cumulative histogram
            if i_tau < n_taus - 1:
                for i in range(dim_out):
                    axs_tau[i_eval_q, 2].fill_between(x=[hist_bins[i] - 0.5, hist_bins[i] + 0.5], y1=pred_cum[i_tau, i],
                                                      y2=pred_cum[i_tau + 1, i], color=colours[i_tau], lw=0)
                    # If ~0 or ~1: plot a line to make the prediction visible
                    if pred[i_tau].cumsum()[i] > 0.9999:
                        axs_tau[i_eval_q, 2].plot([hist_bins[i] - 0.5, hist_bins[i] + 0.5], 2 * [1.0], color=colours[0], lw=2, zorder=3)
                    elif pred[i_tau].cumsum()[i] < 0.0001:
                        axs_tau[i_eval_q, 2].plot([hist_bins[i] - 0.5, hist_bins[i] + 0.5], 2 * [0.0], color=colours[-1], lw=2, zorder=3)

        # Errorbars for leave-one-out quantile range
        if show_leave_one_out:
            y_tau_quantile_min = np.quantile(hist_val_aug_cum, taus[0], axis=0)
            y_tau_quantile_max = np.quantile(hist_val_aug_cum, taus[-1], axis=0)
            y_tau_quantile_median = np.quantile(hist_val_aug_cum, taus[-1], axis=0)
            y_errors = np.vstack([y_tau_quantile_median - y_tau_quantile_min, y_tau_quantile_max - y_tau_quantile_median])
            axs_tau[i_eval_q, 2].errorbar(hist_bins[:-1], y_tau_quantile_median, yerr=y_errors,
                                          color="orange", ls="none", capsize=3, capthick=2, ecolor="orange", zorder=3, lw=2)

        # Additional data
        if add_data is not None:
            axs_tau[i_eval_q, 2].plot(hist_bins[:-1], add_data[i_eval_q, 0, :], ls="none", marker="^", color="white",
                                      mec='k', zorder=5, alpha=0.7)
            axs_tau[i_eval_q, 2].plot(hist_bins[:-1], add_data[i_eval_q, -1, :], ls="none", marker="v", color="white",
                                      mec='k', zorder=5, alpha=0.7)

    plt.subplots_adjust()

    # Adjust the plots
    ind_for_final_place = 2
    for i_eval_q, eval_ind in enumerate(eval_inds):
        # Annotate season and club
        axs_tau[i_eval_q, 2].text(n_clubs, 0.0, seasons_val[eval_ind] + ":\n" + club_names_val[eval_ind], va="bottom", ha="right",
                                  rotation=90, fontsize=16, bbox=dict(facecolor='white', alpha=0.85, edgecolor="black", linewidth=2))
        # Final position
        axs_tau[i_eval_q, ind_for_final_place].text(n_clubs - 3.2, 0, str(int(final_pos_val_resh[eval_ind])), va="bottom", ha="right",
                                                    fontsize=16, color="white",
                                                    bbox=dict(facecolor='k', alpha=1.0, edgecolor="white", linewidth=2,
                                                              boxstyle='round'))
        axs_tau[i_eval_q, ind_for_final_place].axvline(final_pos_val_resh[eval_ind], color="0.5", ls="--", ymin=0, ymax=1)
        # Set limits
        axs_tau[i_eval_q, 1].set_ylim([-0.1, 1.1])
        axs_tau[i_eval_q, 2].set_ylim([-0.1, 1.1])
        axs_tau[i_eval_q, 1].set_xticks(np.arange(1, n_clubs + 1, 1))
        axs_tau[i_eval_q, 2].set_xticks(np.arange(1, n_clubs + 1, 1))

    # Labels for cumulative plot
    axs_tau[n_eval_q - 1, 2].set_xlabel("Position", fontdict={"size": 12})
    axs_tau[(n_eval_q - 1) // 2, 2].set_ylabel("Cumulative histogram", fontdict={"size": 12})
    plt.subplots_adjust(wspace=0.2, hspace=0.0)

    [axs_tau[i_eval_q, 2].set_xlabel(axs_tau[i_eval_q, 2].get_xlabel(), fontdict={"size": 14}) for i_eval_q in range(3)]
    [axs_tau[i_eval_q, 2].set_ylabel(axs_tau[i_eval_q, 2].get_ylabel(), fontdict={"size": 14}) for i_eval_q in range(3)]
    [axs_tau[i_eval_q, 2].set_xticklabels(axs_tau[i_eval_q, 2].get_xticklabels(), fontdict={"size": 9.5}) for i_eval_q in range(3)]
    [axs_tau[i_eval_q, 2].set_yticklabels(axs_tau[i_eval_q, 2].get_yticklabels(), fontdict={"size": 9.5}) for i_eval_q in range(3)]

    fig_tau.savefig(os.path.join(model_path, "examples.pdf"), bbox_inches="tight")
    plt.close(fig_tau)

    # Save predictions
    all_preds_cum = np.asarray(all_preds_cum)
    np.save(os.path.join(model_path, "all_preds_cum"), all_preds_cum)
