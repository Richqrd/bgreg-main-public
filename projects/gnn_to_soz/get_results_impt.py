import os
import pickle
from pathlib import Path
import pandas as pd
from scipy import stats
from bgreg.native.datapaths import datarecord_path
import numpy as np
from bgreg.data.analysis.graph.graph_representation import load_data
import json
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

def get_data(subject, log_path="logs", adj_length=None, feature=None, node_feat="ones", edge_feat="ones", adj="coh",
             link_cutoff=0.3, classify="binary", comb_node_feat=None, comb_edge_feat=None, combined_features = False,
             logdir=None):
    """
    Reads GNN performance metric data from log directories
    :param subject: string, subject ID to pull data from
    :return: DataFrame of performance metrics
    """

    subject = subject

    # create paths for regular gres
    logsdir = Path(datarecord_path, "logs")
    logsdir = Path(logsdir, classify)
    logsdir = Path(logsdir, subject)
    if logdir is None:
        if not combined_features:
            logdir = Path(logsdir,
                          "nf-" + node_feat +
                          "_ef-" + edge_feat +
                          "_adj-" + adj +
                          "_link-" + str(link_cutoff))
        else:
            logdir = Path(logsdir,
                          "cnf-" + "_".join(comb_node_feat) +
                          "_cef-" + "_".join(comb_edge_feat) +
                          "_adj-" + adj +
                          "_link-" + str(link_cutoff))
    logdir = Path(str(logdir).replace("logs", log_path))

    data, ch_names, preictal_size, ictal_size, postictal_size = load_data(subject, node_f=node_feat, edge_f=edge_feat,
                                                                          adj_type=adj, normalize=True,
                                                                          link_cutoff=link_cutoff,
                                                                          classification=classify,
                                                                          combined_features=combined_features,
                                                                          comb_node_feat=comb_node_feat,
                                                                          comb_edge_feat=comb_edge_feat,
                                                                          self_loops=False)


    result_df = pd.DataFrame(columns=['subject', 'soz', 'nonsoz'])
    soz_importances = []
    nonsoz_importances = []

    # load data
    with open(os.path.join(logdir, "ictal_imptnce_scores.pickle"), 'rb') as file:
        with open(os.path.join("projects/gnn_to_soz", "soz_contacts.json"), 'r') as f:
            soz_contacts_json = json.load(f)
            soz_contacts = soz_contacts_json[subject]['contacts']

            importances = pickle.load(file)

            # normalize for each attention signal to make score between 0 and 1
            scores = [(impt - impt.min()) / (impt.max() - impt.min()) for impt in importances]
            scores = np.array(scores)

            mean_score = scores.mean(0)

            top_n = 1000
            top_indices = get_topn_indices(mean_score, top_n)

            soz_num = 0
            nonsoz_num = 0
            for index in top_indices:
                if ch_names[index] in soz_contacts:
                    soz_importances.extend(list(np.transpose(scores)[index]))
                    soz_num += 1
                else:
                    nonsoz_importances.extend(list(np.transpose(scores)[index]))
                    nonsoz_num += 1

            new_row = {
                'subject': subject,
                'soz': soz_num,
                'nonsoz': nonsoz_num
            }
            new_row = pd.DataFrame([new_row])
            result_df = pd.concat([result_df, new_row], ignore_index=True)


    return soz_importances, nonsoz_importances, result_df

def get_topn_indices(lst,n):
    # Get the indices of the sorted list in descending order
    sorted_indices = sorted(range(len(lst)), key=lambda x: lst[x], reverse=True)
    # Return the first three indices
    return sorted_indices[:n]



def boxplots(df, log_path):
    # set font and figures params
    plt.rcParams["font.size"] = 24
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Arial"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (7, 5)
    color_palette = ["firebrick", "#0697A7"]

    fig, ax = plt.subplots()
    flierprops = dict(marker='x', markerfacecolor='black', markersize=12,  linestyle='none', alpha=1)
    props = {
        'boxprops': {'edgecolor': 'black'},
        'medianprops': {'color': 'white'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }

    kws = {"linewidth": 1.0}
    x = 'label'
    y = 'importance'
    # sns.stripplot(data=df, x=x,y=y, palette=color_palette, edgecolor="black", **kws)
    box = sns.boxplot(data=df, x=x, y=y, notch=False, palette=color_palette, saturation=0.8, width=0.9,
                flierprops=flierprops, **props)

    pairs = [('SOZ+','SOZ-')]
    annotator = Annotator(box, pairs, data=df, x=x, y=y)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()


    metric = 'Importance'
    ax.set_xlabel('Electrode Region')
    ax.set_ylabel(metric)
    # ax.set_ylim(0, 1.02)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.2)

    ytick_positions = range(0, 110, 20)
    ytick_positions = [n / 100 for n in ytick_positions]
    ytick_labels = [str(y) for y in ytick_positions]  # Convert positions to strings for labeling
    plt.yticks(ytick_positions, ytick_labels)

    # ytick_positions = range(0, 300, 50)
    # ytick_labels = [str(y) for y in ytick_positions]  # Convert positions to strings for labeling
    # plt.yticks(ytick_positions, ytick_labels)

    filename = metric + "." + 'pdf'
    savefile = os.path.join(datarecord_path, log_path, filename)
    # plt.savefig(savefile, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(savefile)


if __name__ == "__main__":
    log_path = "logs_new_spectrum_eccgat"

    # list of subjects to pull logs from
    subjects = ['ummc002', 'ummc003', 'ummc004', 'ummc005', 'ummc006', 'ummc009', 'pt01', 'pt2', 'pt3', 'pt8', 'pt13', 'pt16', 'umf001']
    # subjects = ['ummc002', 'ummc003']

    soz_importances = []
    nonsoz_importances = []
    overall_df = pd.DataFrame(columns=['subject', 'soz', 'nonsoz'])

    for subject in subjects:
        soz, nonsoz, df = get_data(subject, log_path=log_path,
                                node_feat="ones", edge_feat="ones", adj="coh",
                                combined_features=True, comb_node_feat=['energy', 'band_energy'], comb_edge_feat=['corr', 'coh'],
                                classify="binary")
        overall_df = pd.concat([overall_df, df], ignore_index=True)
        soz_importances.extend(soz)
        nonsoz_importances.extend(nonsoz)

    stat, p = stats.ranksums(soz_importances, nonsoz_importances)
    print(p)

    overall_df.to_csv(os.path.join(datarecord_path, log_path,'top_importances.csv'))
    print(soz_importances)
    print(nonsoz_importances)

    # Creating a DataFrame for x
    df_x = pd.DataFrame({'label': ['SOZ+'] * len(soz_importances), 'importance': soz_importances})

    # Creating a DataFrame for y
    df_y = pd.DataFrame({'label': ['SOZ-'] * len(nonsoz_importances), 'importance': nonsoz_importances})

    # Concatenating the two DataFrames
    importance_df = pd.concat([df_x, df_y], ignore_index=True)
    importance_df.to_csv(os.path.join(datarecord_path, log_path, 'all_importances.csv'))

    boxplots(importance_df, log_path)

    # find index of electrodes with highest importance scores
    # get their importance scores
    # label them as soz or non-soz electrodes
    # for each participant, create a dataframe explaining how many soz or non soz electrodes

    # create a list of importance scores of SOZ electrodes
    # create a list of importance socres of non SOZ electrodes
    # stat test

    # this function will return an soz list, non soz list, dataframe row of participant
    # concatenate lists and dataframes




