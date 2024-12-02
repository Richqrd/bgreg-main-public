import os
import pickle
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# silence matplotlib warning on max number of figures
matplotlib.rcParams.update({'figure.max_open_warning': 0})


def add_stats(t_dict, subject, filepath):
    """
    Loads statistics from pickle files
    :param t_dict: dict, dictionary of test to read from
    :param subject: string, subject ID
    :param filepath: string, subject/filepath
    :return:
    """
    test_filename = "test_stats.pickle"
    train_val_filename = "train_val_stats.pickle"
    # Not using attention scores, yet.
    # attention_scores_filename = "attention_scores.pickle"

    t_dict[subject] = {"test": None, "train_val": None}

    # update test statistics
    with open(os.path.join(filepath, test_filename), "rb") as f:
        t_dict[subject]["test"] = pickle.load(f)

    # update train val statistics
    with open(os.path.join(filepath, train_val_filename), "rb") as f:
        t_dict[subject]["train_val"] = pickle.load(f)


def get_stats_dicts(test_type="binary"):
    """
    Create a dictionary with the statistics collected from all tests,
    organized as follows:
    {"b1": b1_dict, "b2": b2_dict, "b3": b3_dict,
            "t11": t11_dict, "t12": t12_dict, "t13": t13_dict,
            "t21": t21_dict, "t22": t22_dict, "t23": t23_dict}

    bX entries refer to the baseline results for the 3 graph representations.
    t1X entries refer to the test1 results for the 3 graph representations.
    t2X entries refer to the test2 results for the 3 graph representations.

    Graph representations per test were organized as follows:
    b1, {adj: corr, nf: ones, ef: ones}
    b2, {adj: phase, nf: ones, ef: ones}
    b3, {adj: coh, nf: ones, ef: ones}

    t11, {adj: corr, nf: energy, ef: phase}
    t12, {adj: phase, nf: energy, ef: corr}
    t13, {adj: coh, nf: energy, ef: phase}

    t21, {adj: corr, nf: [energy, band_energy], ef: [coh, phase]}
    t22, {adj: phase, nf: [energy, band_energy], ef: [coh, phase]}
    t23, {adj: coh, nf: [energy, band_energy], ef: [coh, phase]}

    These 9 objects are stacked in a dictionary and returned.

    :return: dictionary with statistics per test
    """
    # Fixed directory label
    root_dir = Path(os.getcwd(), "brain_greg", "projects", "gnn_to_soz", "logs", test_type)

    b1_dir = "nf-ones_ef-ones_adj-corr_link-0.3"
    b2_dir = "nf-ones_ef-ones_adj-phase_link-0.3"
    b3_dir = "nf-ones_ef-ones_adj-coh_link-0.3"

    t11_dir = "nf-energy_ef-phase_adj-corr_link-0.3"
    t12_dir = "nf-energy_ef-corr_adj-phase_link-0.3"
    t13_dir = "nf-energy_ef-phase_adj-coh_link-0.3"

    t21_dir = "cnf-energy_band_energy_cef-coh_phase_adj-corr_link-0.3"
    t22_dir = "cnf-energy_band_energy_cef-corr_coh_adj-phase_link-0.3"
    t23_dir = "cnf-energy_band_energy_cef-corr_phase_adj-coh_link-0.3"

    # Dictionaries to store test outcome statistics
    # bX_dict are for the baseline models
    b1_dict = {}
    b2_dict = {}
    b3_dict = {}

    # t1X_dict are for the test1 (nf:energy, ef:{corr-coh-phase}) models
    t11_dict = {}
    t12_dict = {}
    t13_dict = {}

    # t2X_dict are for the test2 (nf:energy, band-energy, ef:{corr-coh-phase}) models
    t21_dict = {}
    t22_dict = {}
    t23_dict = {}

    for subject_dir in os.listdir(os.path.join(os.getcwd(), "brain_greg", root_dir)):
        if ".DS_Store" in subject_dir or subject_dir == "jh102":
            # Not using jh102, yet
            continue

        for test_dir in os.listdir(os.path.join(root_dir, subject_dir)):

            if ".DS_Store" in test_dir:
                continue

            if "nf-ones" in test_dir:
                # reading from baseline directories
                if test_dir == b1_dir:
                    add_stats(b1_dict, subject_dir, Path(root_dir, subject_dir, test_dir))
                if test_dir == b2_dir:
                    add_stats(b2_dict, subject_dir, Path(root_dir, subject_dir, test_dir))
                if test_dir == b3_dir:
                    add_stats(b3_dict, subject_dir, Path(root_dir, subject_dir, test_dir))

            if "nf-energy" in test_dir:
                # reading from test1 directories
                if test_dir == t11_dir:
                    add_stats(t11_dict, subject_dir, Path(root_dir, subject_dir, test_dir))
                if test_dir == t12_dir:
                    add_stats(t12_dict, subject_dir, Path(root_dir, subject_dir, test_dir))
                if test_dir == t13_dir:
                    add_stats(t13_dict, subject_dir, Path(root_dir, subject_dir, test_dir))

            if "cnf" in test_dir:
                # reading from test2 directories
                if test_dir == t21_dir:
                    add_stats(t21_dict, subject_dir, Path(root_dir, subject_dir, test_dir))
                if test_dir == t22_dir:
                    add_stats(t22_dict, subject_dir, Path(root_dir, subject_dir, test_dir))
                if test_dir == t23_dir:
                    add_stats(t23_dict, subject_dir, Path(root_dir, subject_dir, test_dir))

    return {"b1": b1_dict, "b2": b2_dict, "b3": b3_dict,
            "t11": t11_dict, "t12": t12_dict, "t13": t13_dict,
            "t21": t21_dict, "t22": t22_dict, "t23": t23_dict}


def isdir(dir_to_check):
    if not os.path.exists(dir_to_check):
        os.mkdir(dir_to_check)


def get_metrics_list(t_dict, test_or_train, metric):
    """
    Get metrics from test as a list of values
    :param t_dict: dict, dictionary of test to read from
    :param test_or_train: string, "test" or "train_val"
    :param metric: string, metric to read from
    :return: list of metrics
    """
    metrics_list = []
    for subject in t_dict.keys():
        metrics_list.append(t_dict[subject][test_or_train][metric])
    return metrics_list


def get_metrics_dict(t_dict, test_or_train, metric):
    """
    Get metrics from test as a dictionary of values using subjects as keys
    :param t_dict: dict, dictionary of test to read from
    :param test_or_train: string, "test" or "train_val"
    :param metric: string, metric to read from
    :return: dictionary of metrics
    """
    metrics_dict = {}
    for subject in t_dict.keys():
        if test_or_train == "train_val":
            metrics_dict[subject] = np.mean(t_dict[subject][test_or_train][metric])
        else:
            metrics_dict[subject] = t_dict[subject][test_or_train][metric]
    return metrics_dict


def get_df_metrics(test_or_train, metric):
    """
    Create dataframe with metrics by username as rows and test type (baseline, test1, test2) as columns
    df = pd.DataFrame({"b1": b1_metrics, "b2": b2_metrics, "b3": b3_metrics,
                       "t11": t11_metrics, "t12": t12_metrics, "t13": t13_metrics,
                       "t21": t21_metrics, "t22": t22_metrics, "t23": t23_metrics})
    the {tt}_metrics objects here are dictionaries with the test ({tt}) identifier as keys
    and a float value or list of metrics, explained below.
        
    When test_or_train is "test", the metrics are a float value corresponding to 
    the metric computed by averaging the outputs of the GNN model test evaluation.

    When test_or_train is "train_vale", the metrics are lists of float values corresponding to
    the metric computed at each stage (epoch) of the GNN model train/validation evaluation.
    
    :param test_or_train: string, "test" or "train_val"
    :param metric: string, metric to read from
    :return: dataframe
    """
    df_dict = {}
    for t_id, t_dict in stats_dict.items():
        df_dict[t_id] = get_metrics_dict(t_dict, test_or_train, metric)
    return pd.DataFrame(df_dict)


def save_boxplot(df, metric, stats_dir=Path(os.getcwd(), "brain_greg", "projects", "gnn_to_soz", "statistics"),
                 test_type="binary", file_fmt="pdf"):
    """
    Save box plots of statistics into png files
    :param df: DataFrame, dataframe with statistics
    :param metric: string, metric label, i.e., Test F1-Score
    :param stats_dir: string, directory where to save boxplots
    :param test_type: string, label of test type i.e., "binary", "multi"
    :param file_fmt: string, file format to save boxplots i.e., "pdf", "png"
    :return:
    """
    # set font and figures params
    plt.rcParams["font.size"] = 24
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Arial"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (7, 5)
    color_palette = "husl"

    fig, ax = plt.subplots()
    flierprops = dict(marker="o")
    props = {
        'boxprops': {'edgecolor': 'black'},
        'medianprops': {'color': 'white'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }
    sns.boxplot(data=df, notch=False, palette=color_palette, saturation=0.8, width=0.9, flierprops=flierprops, **props)
    kws = {"linewidth": 1.0}
    sns.stripplot(data=df, palette=color_palette, edgecolor="black", **kws)

    ax.set_xlabel('GR Combination Test')
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.5)
    filename = metric + "." + file_fmt
    savedir = Path(stats_dir, test_type)
    isdir(savedir)
    savefile = Path(savedir, filename)
    plt.savefig(savefile, bbox_inches='tight', pad_inches=0.1)


def save_boxplot_patient_cluster(df, metric, test_type="binary"):
    """
    Save boxplots by clusters of patients (UMMC, JH, PT)
    :param df: DataFrame, dataframe with statistics
    :param metric: string, metric label, i.e., Test F1-Score
    :param test_type: string, label of test type i.e., "binary", "multi"
    :return:
    """
    # FIXME: find a better way of creating this logging directory elsewhere
    savedir = Path(os.getcwd(), "brain_greg", "projects", "gnn_to_soz", "statistics", test_type)
    isdir(savedir)
    savedir = Path(savedir, "patient-clusters")
    isdir(savedir)

    ummc_metric = metric + " UMMC"
    isdir(Path(savedir, "ummc"))
    save_boxplot(df.filter(like='ummc', axis=0), ummc_metric,
                 stats_dir=Path(savedir, "ummc"))

    jh_metric = metric + " JH"
    isdir(Path(savedir, "jh"))
    save_boxplot(df.filter(like='jh', axis=0), jh_metric,
                 stats_dir=Path(savedir, "jh"))

    pt_metric = metric + " PT"
    isdir(Path(savedir, "pt"))
    save_boxplot(df.filter(like='pt', axis=0), pt_metric,
                 stats_dir=Path(savedir))


test_t = "multi-normal"
# Get dictionary with statistics (see get_stats_dicts() function above)
stats_dict = get_stats_dicts(test_t)
# Get test, train, and val statistics as list of tuples (df, str_metric)
dfs = [(get_df_metrics("test", "test_loss").dropna(axis=0), "Test Loss"),
       (get_df_metrics("test", "test_acc").dropna(axis=0), "Test Accuracy"),
       (get_df_metrics("test", "test_auc").dropna(axis=0), "Test AUC"),
       (get_df_metrics("test", "test_f1_score").dropna(axis=0), "Test F1-Score"),
       (get_df_metrics("train_val", "train_loss").dropna(axis=0), "Train Loss"),
       (get_df_metrics("train_val", "train_acc").dropna(axis=0), "Train Accuracy"),
       (get_df_metrics("train_val", "train_auc").dropna(axis=0), "Train AUC"),
       (get_df_metrics("train_val", "train_f1_score").dropna(axis=0), "Train F1-Score"),
       (get_df_metrics("train_val", "val_loss").dropna(axis=0), "Val Loss"),
       (get_df_metrics("train_val", "val_acc").dropna(axis=0), "Val Accuracy"),
       (get_df_metrics("train_val", "val_auc").dropna(axis=0), "Val AUC"),
       (get_df_metrics("train_val", "val_f1_score").dropna(axis=0), "Val F1-Score")]

# Save box plot statistics into png files at path projects/gnn_to_soz/statistics
for df_metric in dfs:
    save_boxplot(df_metric[0], df_metric[1], test_type=test_t)
    save_boxplot_patient_cluster(df_metric[0], df_metric[1], test_type=test_t)

# test_scores = [dfs[1], dfs[2], dfs[3]]
# t21 = "t21"
# t22 = "t22"
# t23 = "t23"
# for tscore in test_scores:
#     print(np.mean(tscore[0][t21]))
# tmp = 0
