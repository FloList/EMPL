"""
This script generates the training and test data from the match results saved in Bundesliga_Results.csv
(needs to be downloaded from https://www.kaggle.com/thefc17/bundesliga-results-19932018).
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_data_season(seas, do_weeks=True):
    """
    This function computes the points, goals scored and taken, for a single season.
    :param seas: string containing the season
    :param do_weeks: if True: resulting table will be n_matches x n_clubs (new row only when ALL clubs have played the next match)
                     if False: resulting table will be n_dates x n_clubs (new row when a match takes place)
    :return: points, goals scored, goals taken, team_names
    """
    # Extract the season
    data_seas = dataset[dataset["Season"] == seas]
    if data_seas.shape[0] == 0:
        print("EMPTY DATASET!")
        return
    # Get the number of matches and clubs
    n_matches = data_seas.shape[0]
    unique_clubs = np.unique(np.asarray(data_seas.get(["HomeTeam", "AwayTeam"])).flatten())
    n_clubs = len(unique_clubs)
    # Make an array of match dates
    dates_raw = np.asarray(data_seas.get("Date"))
    dates = np.asarray([date.split("/") for date in dates_raw]).astype(int)
    # Set 93, 94, ... to 1993, 1994
    for i_year, year in enumerate(dates[:, -1]):
        if year < 100:
            if dates[i_year, -1] < 21:  # 20th century
                dates[i_year, -1] += 2000
            elif dates[i_year, -1] < 100:  # 19th century
                dates[i_year, -1] += 1900
    unique_dates = np.unique(dates, axis=0).T
    n_dates = unique_dates.shape[1]
    # Sort the dates by year, month, and day
    unique_dates = np.array(list(zip(unique_dates[0], unique_dates[1], unique_dates[2])) , dtype=[("day", int), ("month", int), ("year", int)])
    unique_dates.sort(order=("year", "month", "day"))
    unique_dates = unique_dates.view(int).reshape([-1, 3])
    # Get club IDs of home and away matches
    home_ID = np.asarray([np.argwhere(team == unique_clubs).flatten() for team in data_seas["HomeTeam"]]).flatten()
    away_ID = np.asarray([np.argwhere(team == unique_clubs).flatten() for team in data_seas["AwayTeam"]]).flatten()
    # Initialise arrays
    played_team = np.zeros((n_dates, n_clubs), dtype=bool)
    points_team = np.zeros((n_dates, n_clubs), dtype=np.int32)
    goals_scored_team = np.zeros((n_dates, n_clubs), dtype=np.int32)
    goals_taken_team = np.zeros((n_dates, n_clubs), dtype=np.int32)
    # Number of points for a win
    points_win = 3 if int(seas[:4]) >= 1995 else 2  # change in counting: 2 points for win -> 3 points for win

    # Iterate over the matches and fill the data for the dates
    for i_match in range(n_matches):
        glob_ind = data_seas.index[i_match]
        this_date = dates[i_match]
        date_ind = np.argwhere((this_date[0] == unique_dates[:, 0]) & (this_date[1] == unique_dates[:, 1]) & (this_date[2] == unique_dates[:, 2])).flatten()[0]
        goals_scored_team[date_ind, home_ID[i_match]] = data_seas["FTHG"][glob_ind]
        goals_scored_team[date_ind, away_ID[i_match]] = data_seas["FTAG"][glob_ind]
        goals_taken_team[date_ind, home_ID[i_match]] = data_seas["FTAG"][glob_ind]
        goals_taken_team[date_ind, away_ID[i_match]] = data_seas["FTHG"][glob_ind]
        played_team[date_ind, home_ID[i_match]] = True
        played_team[date_ind, away_ID[i_match]] = True

        # Home wins
        if data_seas["FTHG"][glob_ind] > data_seas["FTAG"][glob_ind]:
            points_team[date_ind, home_ID[i_match]] = points_win
        # Away wins
        elif data_seas["FTHG"][glob_ind] < data_seas["FTAG"][glob_ind]:
            points_team[date_ind, away_ID[i_match]] = points_win
        # Draw
        else:
            points_team[date_ind, home_ID[i_match]] = 1
            points_team[date_ind, away_ID[i_match]] = 1

    # Do weeks (n_dates -> n_matches)
    if do_weeks:
        n_matches_till_date = played_team.cumsum(0)
        n_matches_per_team = 2 * (n_clubs - 1)

        points_team_week = np.zeros((n_matches_per_team, n_clubs))
        goals_scored_team_week = np.zeros((n_matches_per_team, n_clubs))
        goals_taken_team_week = np.zeros((n_matches_per_team, n_clubs))

        # Iterate over the weeks and collect the playing dates for each club
        for i_week in range(1, n_matches_per_team + 1):
            for i_club in range(n_clubs):
                playing_dates_ind = np.min(np.argwhere(n_matches_till_date[:, i_club] == i_week).flatten())  # min: after this playing date, club did not play in this week
                points_team_week[i_week-1, i_club] = points_team[playing_dates_ind, i_club]
                goals_scored_team_week[i_week-1, i_club] = goals_scored_team[playing_dates_ind, i_club]
                goals_taken_team_week[i_week-1, i_club] = goals_taken_team[playing_dates_ind, i_club]

        assert np.all(points_team.sum(0) == points_team_week.sum(0)), "Total points were not calculated correctly!"
        assert np.all(goals_scored_team.sum(0) == goals_scored_team_week.sum(0)), "Scored goals were not calculated correctly!"
        assert np.all(goals_taken_team.sum(0) == goals_taken_team_week.sum(0)), "Taken goals were not calculated correctly!"

    # Continue with the desired arrays
    points_team_out = points_team if not do_weeks else points_team_week
    goals_scored_team_out = goals_scored_team if not do_weeks else goals_scored_team_week
    goals_taken_team_out = goals_taken_team if not do_weeks else goals_taken_team_week

    return points_team_out, goals_scored_team_out, goals_taken_team_out, unique_clubs


def make_table(points, goals_scored, goals_taken):
    """
    This function takes arrays of points, goals scored and taken in each match of a season and returns the league table.
    :param points: points
    :param goals_scored: goals scored
    :param goals_taken: goals taken
    :return: table (n_weeks x n_clubs)
    """
    n_out, n_clubs = points.shape

    # Make table of standings
    table = np.zeros((n_out, n_clubs))
    cum_points = np.cumsum(points, axis=0)
    cum_goals_scored = np.cumsum(goals_scored, axis=0)
    cum_goals_taken = np.cumsum(goals_taken, axis=0)
    cum_goals_diff = cum_goals_scored - cum_goals_taken

    # Iterate over the weeks / dates
    for i in range(n_out):
        concat_array = np.concatenate([cum_points[i:i+1],
                                       cum_goals_diff[i:i+1],
                                       cum_goals_scored[i:i+1]], axis=0).T
        cum_array_neg = np.array([tuple(-concat_array[j]) for j in range(concat_array.shape[0])],
                             dtype=[('points', '<i4'), ('diff', '<i4'), ('scored', '<i4')])
        ranking = np.asarray([np.argsort(cum_array_neg, order=('points', 'diff', 'scored'))])  # sort by: 1) points, 2) goal difference, 3) goals scored
        table[i] = [np.argwhere(j == ranking.flatten()).flatten()[0] + 1 for j in range(n_clubs)]  # get position (+1 because leader is 1st, not 0th)

    return table


def print_table(table, clubs, points, goals_scored, goals_taken):
    """
    Function to print the league table.
    :param table: 1D array with positions
    :param clubs: club names
    :param points: points
    :param goals_scored: goals scored
    :param goals_taken: goals taken
    """
    from tabulate import tabulate
    inds = np.argsort(table)
    clubs_sorted = clubs[inds]
    points_sorted = np.asarray(points[inds], dtype=int)
    goals_scored_sorted = np.asarray(goals_scored[inds], dtype=int)
    goals_taken_sorted = np.asarray(goals_taken[inds], dtype=int)
    goals_diff_sorted = goals_scored_sorted - goals_taken_sorted
    pos = np.arange(1, len(inds) + 1)

    tab_all = tabulate(list(zip(pos, clubs_sorted, points_sorted, goals_scored_sorted, goals_taken_sorted, goals_diff_sorted)),
                       headers=["#", "Club", "Points", "Goals scored", "Goals taken", "Diff."])

    print(tab_all)

if __name__ == "__main__":
    csv_url = "https://www.kaggle.com/thefc17/bundesliga-results-19932018"
    csv_path = "./Bundesliga_Results.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("File 'Bundesliga_Results.csv' not found in this folder! Please download it from " + csv_url)

    # Read dataset
    dataset = pd.read_csv(csv_path)
    all_results = np.asarray(dataset.get(["FTHG", "FTAG"]))
    seasons = np.asarray(dataset.get("Season"))
    n_seasons = len(np.unique(seasons))
    print("Number of seasons:", n_seasons)
    dates_raw = np.asarray(dataset.get("Date"))
    dates = np.asarray([date.split("/") for date in dates_raw]).astype(int)
    all_clubs = np.asarray(dataset.get(["HomeTeam", "AwayTeam"]))
    assert np.all(np.unique(all_clubs[:, 0]) == np.unique(all_clubs[:, 1])), "Home and away clubs don't match!"
    n_clubs_total = len(np.unique(all_clubs.flatten()))
    print("Number of clubs:", n_clubs_total)
    n_matches_total = dataset.shape[0]
    print("Total number of matches", n_matches_total)

    # Define seasons
    years = np.arange(1995, 2018)  # 1993/94 and 1994/95: only 2 points for a win (might affect histograms)
    seasons = [str(y) + "-" + str(int(y+1))[-2:].zfill(2) for y in years]

    # Load data (note: penalty points e.g. for Dynamo Dresden in 93/94 are not counted here!)
    table_all = []
    points_all = []
    goals_scored_all = []
    goals_taken_all = []
    club_names_all = []

    for season in seasons:
        print("Season", season)
        points, goals_scored, goals_taken, club_names = get_data_season(season, do_weeks=True)
        points_all.append(points)
        goals_scored_all.append(goals_scored)
        goals_taken_all.append(goals_taken)
        club_names_all.append(club_names)

        table_all.append(make_table(points, goals_scored, goals_taken))

        print("Final standings:")
        print_table(table_all[-1][-1, :], club_names, points.sum(0), goals_scored.sum(0), goals_taken.sum(0))
        print("\n\n")

    # Convert to numpy arrays
    club_names_all = np.asarray(club_names_all)
    table_all = np.asarray(table_all)
    points_all = np.asarray(points_all)
    goals_scored_all = np.asarray(goals_scored_all)
    goals_taken_all = np.asarray(goals_taken_all)
    n_seasons_out = club_names_all.shape[0]

    # Select seasons for validation/testing and for training
    val_indices = [3, 11, 19]
    train_indices = np.setdiff1d(np.arange(n_seasons_out), val_indices)
    print("Years for validation:", years[val_indices])
    print("Years for training:", years[train_indices])
    club_names_val, table_val, points_val, goals_scored_val, goals_taken_val = \
            club_names_all[val_indices], table_all[val_indices], points_all[val_indices], \
            goals_scored_all[val_indices], goals_taken_all[val_indices]
    club_names_train, table_train, points_train, goals_scored_train, goals_taken_train = \
            club_names_all[train_indices], table_all[train_indices], points_all[train_indices], \
            goals_scored_all[train_indices], goals_taken_all[train_indices]

    # Augment training data by re-playing the seasons, randomly shuffling the weeks
    n_seasons_train, n_seasons_val = club_names_train.shape[0], club_names_val.shape[0]
    n_augment = 1000

    # For training data
    print("Doing augmentation for training data by randomly shuffling the weeks...")
    table_aug, points_aug, goals_scored_aug, goals_taken_aug = [], [], [], []
    for i_aug in tqdm(range(n_augment)):
        # Draw a random permutation of the weeks:
        rand_perm = np.random.permutation(table_train.shape[1])
        points_aug.append(points_train[:, rand_perm])
        goals_scored_aug.append(goals_scored_train[:, rand_perm])
        goals_taken_aug.append(goals_taken_train[:, rand_perm])
        table_aug.append([make_table(points_train[i, rand_perm], goals_scored_train[i, rand_perm], goals_taken_train[i, rand_perm])
                          for i in range(n_seasons_train)])

    table_aug, points_aug, goals_scored_aug, goals_taken_aug = \
        np.asarray(table_aug), np.asarray(points_aug), np.asarray(goals_scored_aug), np.asarray(goals_taken_aug)

    # For validation data: this is not needed, but might be useful for additional bootstrapping cross-checks.
    print("Doing augmentation for validation data by randomly shuffling the weeks...")
    table_val_aug, points_val_aug, goals_scored_val_aug, goals_taken_val_aug = [], [], [], []
    for i_aug in tqdm(range(n_augment)):
        # Draw a random permutation of the weeks:
        rand_perm = np.random.permutation(table_val.shape[1])
        points_val_aug.append(points_val[:, rand_perm])
        goals_scored_val_aug.append(goals_scored_val[:, rand_perm])
        goals_taken_val_aug.append(goals_taken_val[:, rand_perm])
        table_val_aug.append([make_table(points_val[i, rand_perm], goals_scored_val[i, rand_perm], goals_taken_val[i, rand_perm]) for i in range(n_seasons_val)])

    table_val_aug, points_val_aug, goals_scored_val_aug, goals_taken_val_aug = \
        np.asarray(table_val_aug), np.asarray(points_val_aug), np.asarray(goals_scored_val_aug), np.asarray(goals_taken_val_aug)

    # Save / load data
    output_folder = "."
    output_file = os.path.join(output_folder, "train_data")
    output_file_val = os.path.join(output_folder, "val_data")
    output_file_val_aug = os.path.join(output_folder, "val_data_augmented")

    # Save data
    np.savez(output_file, table=table_aug, club_names=club_names_train, points=points_aug,
             goals_scored=goals_scored_aug, goals_taken=goals_taken_aug)
    np.savez(output_file_val, table=table_val, club_names=club_names_val, points=points_val,
             goals_scored=goals_scored_val, goals_taken=goals_taken_val)
    np.savez(output_file_val_aug, table=table_val_aug, club_names=club_names_val, points=points_val_aug,
             goals_scored=goals_scored_val_aug, goals_taken=goals_taken_val_aug)

    # Load data
    # data_train = np.load(output_file + ".npz", allow_pickle=True)
    # data_val = np.load(output_file_val + ".npz", allow_pickle=True)
    # print(data_train.files)
    # table_aug, club_names_aug, points_aug, goals_scored_aug, goals_taken_aug = \
    #     data_train["table"], data_train["club_names"], data_train["points"], data_train["goals_scored"], data_train["goals_taken"]
    # table_val, club_names_val, points_val, goals_scored_val, goals_taken_val = \
    #     data_val["table"], data_val["club_names"], data_val["points"], data_val["goals_scored"], data_val["goals_taken"]
    # n_augment = table_aug.shape[0]

    # Print some CDF quantiles
    import colorcet as cc
    quantiles = np.linspace(0.1, 0.9, 9)
    n_quantiles = len(quantiles)
    i_season_plot = -1
    i_club_plots = [1, 4, 11]  # for i_season_plot = -1:    1: Bayern (1st), 11: Gladback (9th), 4: Koeln (18th)
    colors = cc.cm.bkr(np.linspace(0, 1, n_quantiles))[::-1]
    linestyles = ["-", ":", "-."]

    # Make a plot for some clubs
    fig, axs = plt.subplots(1, 2)
    for i, i_club_plot in enumerate(i_club_plots):
        hist_aug = np.asarray(
            [np.histogram(table_aug[i, i_season_plot, :, i_club_plot], bins=np.arange(1, table_aug.shape[3] + 2), density=True)[0]
             for i in range(n_augment)])
        hist_aug_cum = hist_aug.cumsum(1)
        hist_goals = np.histogram(goals_scored_aug[2, i_season_plot, :, i_club_plot], bins=np.arange(0, 10), density=True)[0]

        # Plot a histogram of the goals in the first axis
        axs[0].plot(np.arange(1, hist_goals.shape[-1] + 1), hist_goals, ls=linestyles[i])

        # Plot the distribution of the cumulative table position histogram in the second axis
        for i_quant, quant in enumerate(quantiles):
            CDF_quant = np.quantile(hist_aug_cum, quant, axis=0)
            axs[1].plot(np.arange(1, hist_aug.shape[-1] + 1), CDF_quant, color=colors[i_quant], ls=linestyles[i])

