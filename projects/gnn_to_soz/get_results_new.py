import os
import pickle
from pathlib import Path
import pandas as pd
from scipy import stats
from bgreg.native.datapaths import datarecord_path

import numpy as np

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




    # load data
    with open(os.path.join(logdir, "train_val_stats.pickle"), 'rb') as file:
        reg_ones = pickle.load(file)

        # print(reg_ones['val_confusion'])

        with open(os.path.join(logdir, "test_stats.pickle"), 'rb') as test_file:
            test_stats = pickle.load(test_file)

            test_tp = np.mean([x['tp'] for x in test_stats['test_confusion']])
            test_fp = np.mean([x['fp'] for x in test_stats['test_confusion']])
            test_tn = np.mean([x['tn'] for x in test_stats['test_confusion']])
            test_fn = np.mean([x['fn'] for x in test_stats['test_confusion']])

            print([x['tp'] for x in test_stats['test_confusion']])

            new_row = {
                'subject': subject,
                'adj_length': adj_length,
                'feature': feature,

                'train_loss': reg_ones['train_loss'][-1],
                'train_acc': reg_ones['train_acc'][-1],
                'train_auc': reg_ones['train_auc'][-1],
                'train_f1_score': reg_ones['train_f1_score'][-1],

                'train_tp': reg_ones['train_confusion'][-1]['tp'],
                'train_fp': reg_ones['train_confusion'][-1]['fp'],
                'train_tn': reg_ones['train_confusion'][-1]['tn'],
                'train_fn': reg_ones['train_confusion'][-1]['fn'],

                'val_loss': reg_ones['val_loss'][-1],
                'val_acc': reg_ones['val_acc'][-1],
                'val_auc': reg_ones['val_auc'][-1],
                'val_f1_score': reg_ones['val_f1_score'][-1],

                'val_tp': reg_ones['val_confusion'][-1]['tp'],
                'val_fp': reg_ones['val_confusion'][-1]['fp'],
                'val_tn': reg_ones['val_confusion'][-1]['tn'],
                'val_fn': reg_ones['val_confusion'][-1]['fn'],

                'computation_time': reg_ones['computation_time'][-1],

                'test_loss': test_stats['test_loss'],
                'test_acc': test_stats['test_acc'],
                'test_auc': test_stats['test_auc'],
                'test_f1_score': test_stats['test_f1_score'],
                'test_computation_time': test_stats['computation_time'],

                'test_tp': test_tp,
                'test_fp': test_fp,
                'test_tn': test_tn,
                'test_fn': test_fn,

            }

            new_row = pd.DataFrame([new_row])

    return new_row

def summary_stats(df):
    metrics = ['test_loss', 'test_acc', 'test_auc', 'test_f1_score', 'computation_time']

    for metric in metrics:
        print(metric + '-'*8)
        mean = round(df[metric].mean(),4)
        sd = round(df[metric].std(),4)
        # print("\makecell[c]{" + str(mean) +"\\\("+ str(sd) +")}")
        print(f"{mean} ({sd})")

if __name__ == "__main__":
    log_path = "logs"

    # list of subjects to pull logs from
    subjects = ["jh101", "jh103", "jh108",
                "pt01", "pt10", "pt12", "pt13", "pt14", "pt16",
                "pt2", "pt3", "pt6", "pt7", "pt8",
                "umf001",
                "ummc001", "ummc002", "ummc003", "ummc004", "ummc005", "ummc006",
                "ummc007", "ummc009"]

    subjects = [
        "ummc002", "ummc003", "ummc004"
    ]

    overall_df = pd.DataFrame()

    for subject in subjects:
        subject_data = get_data(subject, log_path=log_path,
                                node_feat="ones", edge_feat="ones", adj="coh",
                                combined_features=True, comb_node_feat=['energy', 'band_energy'], comb_edge_feat=['corr', 'coh'],
                                classify="node")
        if overall_df.empty:
            overall_df = subject_data
        else:
            overall_df = pd.concat([overall_df, subject_data], ignore_index=True)

    print(overall_df)

    # Calculate False Positive Rate (FPR)
    overall_df['test_fpr'] = overall_df['test_fp'] / (overall_df['test_fp'] + overall_df['test_tn'])

    # Calculate Positive Predictive Value (PPV)
    overall_df['test_ppv'] = overall_df['test_tp'] / (overall_df['test_tp'] + overall_df['test_fp'])

    # Calculate False Positive Rate (FPR)
    overall_df['train_fpr'] = overall_df['train_fp'] / (overall_df['train_fp'] + overall_df['train_tn'])

    # Calculate Positive Predictive Value (PPV)
    overall_df['train_ppv'] = overall_df['train_tp'] / (overall_df['train_tp'] + overall_df['train_fp'])

    summary_stats(overall_df)

    overall_df.to_csv(os.path.join(datarecord_path, log_path,'summary_stats.csv'))



