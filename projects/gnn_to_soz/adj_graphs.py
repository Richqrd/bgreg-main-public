import os
import pickle
from pathlib import Path
import pandas as pd
from scipy import stats
from bgreg.native.datapaths import datarecord_path
import matplotlib.pyplot as plt
from get_results_new import get_data
import seaborn as sns
from statannotations.Annotator import Annotator

def boxplots(df):
    # set font and figures params
    plt.rcParams["font.size"] = 24
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Arial"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (7, 5)
    color_palette = "coolwarm"

    fig, ax = plt.subplots()
    flierprops = dict(marker='x', markerfacecolor='black', markersize=12,  linestyle='none', alpha=1)
    props = {
        'boxprops': {'edgecolor': 'black'},
        'medianprops': {'color': 'white'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }

    kws = {"linewidth": 1.0}
    x = 'adj_length'
    y = 'computation_time'
    sns.stripplot(data=df, x=x,y=y, palette=color_palette, edgecolor="black", **kws)
    box = sns.boxplot(data=df, x=x, y=y, notch=False, palette=color_palette, saturation=0.8, width=0.9,
                flierprops=flierprops, **props)

    pairs = [(1,20)]
    annotator = Annotator(box, pairs, data=df, x=x, y=y)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()


    metric = 'Training Time (s)'
    ax.set_xlabel('Adjacency Matrix Length (s)')
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.02)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.2)

    ytick_positions = range(0, 110, 20)
    ytick_positions = [n / 100 for n in ytick_positions]
    ytick_labels = [str(y) for y in ytick_positions]  # Convert positions to strings for labeling
    plt.yticks(ytick_positions, ytick_labels)

    # ytick_positions = range(0, 300, 50)
    # ytick_labels = [str(y) for y in ytick_positions]  # Convert positions to strings for labeling
    # plt.yticks(ytick_positions, ytick_labels)

    filename = metric + "." + 'pdf'
    savefile = os.path.join(datarecord_path, filename)
    # plt.savefig(savefile, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(savefile)

if __name__ == "__main__":
    log_path = "logs"

    adj_sec = [1,2,5,10,15,20]

    # list of subjects to pull logs from
    subjects = ["jh101", "jh103", "jh108",
                "pt01", "pt10", "pt12", "pt13", "pt14", "pt16",
                "pt2", "pt3", "pt6", "pt7", "pt8",
                "umf001",
                "ummc001", "ummc002", "ummc003", "ummc004", "ummc005", "ummc006",
                "ummc007", "ummc009"]

    overall_df = pd.DataFrame(columns=['subject', 'adj_length', 'feature', 'train_loss', 'train_acc', 'train_auc',
                                      'train_f1_score', 'val_loss', 'val_acc', 'val_auc', 'val_f1_score', 'computation_time',
                                     'test_loss', 'test_acc', 'test_auc', 'test_f1_score', 'test_computation_time'])

    for length in adj_sec:
        log_path = 'logs_' + str(length) + 's'
        for subject in subjects:
            subject_data = get_data(subject, log_path=log_path, feature="ones", adj_length=length)
            overall_df = pd.concat([overall_df, subject_data], ignore_index=True)

    print(overall_df)
    overall_df.to_csv(os.path.join(datarecord_path,'overall_adj.csv'))

    boxplots(overall_df)





