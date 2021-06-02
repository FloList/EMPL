"""
This script generates "leave-one-out" bootstrapping histograms, where a club from the testing dataset competes against
the club from the augmented training dataset in order to estimate the quantiles.
"""
import os
import sys
import numpy as np
from copy import copy
from make_bundesliga_table import make_table

# CHOOSE THE TASK HERE:
#   1: generate leave-one-out histograms for individual seasons/clubs
#   2: combine the results
TASK = 1


def calculate_leave_one_out_hist_with_replacement(train_data, val_data, season_loo, team_loo, n_loo, hist_bins, option=3):
    """
    Calculate "leave-one-out-table" for each validation sample:
    let clubs in validation data "play in different (augmented) seasons" to estimate uncertainties
    :param train_data: training data
    :param val_data: validation data
    :param season_loo: season index for sample for which histograms shall be calculated
    :param team_loo: team index for sample for which histograms shall be calculated
    :param n_loo: number of artificial seasons to consider
    :param hist_bins: histogram bins
    :param option: option for choosing which sample shall be replaced
                   (1: random, 2: same final position, 3: similar # points)
    :return: array of n_loo histograms of the bootstrapping league table
    """
    llo_dict = dict()
    keys = ["points", "goals_scored", "goals_taken", "table"]
    standings_loo = []

    # Copy the dictionary in order to leave the arrays unchanged
    val_d = dict()
    for key in keys:
        val_d[key] = copy(val_data[key])
        llo_dict[key] = val_d[key][season_loo, :, team_loo]

    # Iterate over the number of artificial seasons
    for i_loo in range(n_loo):
        if i_loo % 10 == 0:
            print("i_loo / n_loo:", i_loo, "/", n_loo)
        augment = np.random.choice(train_data["table"].shape[0])
        season = np.random.choice(train_data["table"].shape[1])
        # Option 1) replace random team
        if option == 1:
            replace = np.random.choice(train_data["table"].shape[3])  # replace a random occurrence: problem: "Bayern might need to play against itself"
        # Option 2) replace team that has the same final position (-> similar output)
        elif option == 2:
            final_pos_loo = int(val_d["table"][season_loo, -1, team_loo])
            train_final_table = train_data["table"][augment, season, -1, :]
            replace = np.argwhere(train_final_table == final_pos_loo).flatten()[0]
        # Option 3) replace team with the most similar number of points at the end of the season (-> similar input)
        elif option == 3:
            total_points_scored = val_d["points"][season_loo, :, team_loo].sum()
            train_points_tot = train_data["points"][augment, season, :, :].sum(0)
            replace = np.argmin(np.abs(train_points_tot - total_points_scored)).flatten()[0]
        else:
            raise NotImplementedError

        # Make table
        train_d_loc = dict()
        for key in keys:
            train_d_loc[key] = train_data[key].copy()
            train_d_loc[key][augment, season, :, replace] = llo_dict[key]
        table = make_table(train_d_loc["points"][augment, season, :, :],
                           train_d_loc["goals_scored"][augment, season, :, :],
                           train_d_loc["goals_taken"][augment, season, :, :])
        standings_loo.append(table[:, replace])
    print("Done.")
    standings_hist = np.asarray([np.histogram(np.asarray(standings_loo)[i], bins=hist_bins, density=True)[0] for i in range(n_loo)])
    return standings_hist


# A linear index must be provided as an input such that running this script can be done in parallel:
# This index (this_ind) must be such that:
#     season_loo, team_loo = np.unravel_index(this_ind, (n_seasons_val, n_clubs))
try:
    this_ind = int(sys.argv[1])
except IndexError:
    this_ind = 0

print("Index:", this_ind)

# Load data
output_folder = "."
output_file = os.path.join(output_folder, "train_data")
output_file_val = os.path.join(output_folder, "val_data")
data_train = np.load(output_file + ".npz", allow_pickle=True)
data_val = np.load(output_file_val + ".npz", allow_pickle=True)
print(data_train.files)
n_seasons_val, n_clubs = data_val["club_names"].shape
loo_folder = "Leave_one_out_data"

if TASK == 1:
    # Calculate leave-one-out histograms and save
    n_loo = 200  # number of simulated leave-one-out seasons
    hist_bins = np.arange(1, n_clubs + 2)
    season_loo, team_loo = np.unravel_index(this_ind, (n_seasons_val, n_clubs))
    hist_out = calculate_leave_one_out_hist_with_replacement(data_train, data_val, season_loo, team_loo, n_loo, hist_bins, option=3)
    filename = os.path.join(output_folder, loo_folder, str(this_ind))
    os.makedirs(loo_folder, exist_ok=True)
    np.save(filename, hist_out)
    print("Done with index", this_ind)

elif TASK == 2:
    # Combine the individual files
    all_files = [str(i) + ".npy" for i in range(n_seasons_val * n_clubs)]
    all_hist_data = []
    for file in all_files:
        hist = np.load(os.path.join(output_folder, loo_folder, file))
        all_hist_data.append(hist)
    all_hist_data = np.asarray(all_hist_data)
    all_hist_data = np.transpose(all_hist_data, [1, 0, 2])  # n_loo x n_seasons*n_clubs x n_position
    np.save(os.path.join(output_folder, "leave_one_out_data_all.npy"), all_hist_data)

else:
    raise NotImplementedError
