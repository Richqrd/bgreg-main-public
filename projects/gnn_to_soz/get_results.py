import os
import pickle
from main import create_log_directories
from pathlib import Path
import pandas as pd
from scipy import stats


def get_data(subject):
    """
    Reads GNN performance metric data from log directories
    :param subject: string, subject ID to pull data from
    :return: DataFrame of performance metrics
    """

    subject = subject

    # create paths for regular gres
    combined_features = False
    logdir = create_log_directories(sub=subject, node_feat="ones", edge_feat="ones", adj="corr",
        link_cutoff=0.3, classify="binary", combined_features=combined_features,
        comb_node_feat=None, comb_edge_feat=None, logdir=None)
    combined_features = True
    logdir_cf = create_log_directories(sub=subject, node_feat="ones", edge_feat="ones", adj="corr",
        link_cutoff=0.3, classify="binary", combined_features=combined_features,
        comb_node_feat=["energy", "band_energy"], comb_edge_feat=["coh", "phase"], logdir=None)

    # create paths for gres with extended adj matrix
    adj_logdir = Path(str(logdir).replace("logs", "adj_logs"))
    adj_logdir_cf = Path(str(logdir_cf).replace("logs", "adj_logs"))

    result_df = pd.DataFrame(columns=['subject', 'adj_length', 'feature', 'train_loss', 'train_acc', 'train_auc', 'train_f1_score', 'val_loss', 'val_acc', 'val_auc', 'val_f1_score'])

    # load data
    with open(os.path.join(logdir, "train_val_stats.pickle"), 'rb') as file:
        reg_ones = pickle.load(file)
        new_row = {
            'subject': subject,
            'adj_length': '1s',
            'feature': 'ones',
            'train_loss': reg_ones['train_loss'][-1],
            'train_acc': reg_ones['train_acc'][-1],
            'train_auc': reg_ones['train_auc'][-1],
            'train_f1_score': reg_ones['train_f1_score'][-1],
            'val_loss': reg_ones['val_loss'][-1],
            'val_acc': reg_ones['val_acc'][-1],
            'val_auc': reg_ones['val_auc'][-1],
            'val_f1_score': reg_ones['val_f1_score'][-1],
        }
        new_row = pd.DataFrame([new_row])
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    with open(os.path.join(logdir_cf, "train_val_stats.pickle"), 'rb') as file:
        reg_cf = pickle.load(file)
        new_row = {
            'subject': subject,
            'adj_length': '1s',
            'feature': 'combined',
            'train_loss': reg_cf['train_loss'][-1],
            'train_acc': reg_cf['train_acc'][-1],
            'train_auc': reg_cf['train_auc'][-1],
            'train_f1_score': reg_cf['train_f1_score'][-1],
            'val_loss': reg_cf['val_loss'][-1],
            'val_acc': reg_cf['val_acc'][-1],
            'val_auc': reg_cf['val_auc'][-1],
            'val_f1_score': reg_cf['val_f1_score'][-1],
        }
        new_row = pd.DataFrame([new_row])
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    with open(os.path.join(adj_logdir, "train_val_stats.pickle"), 'rb') as file:
        adj_ones = pickle.load(file)
        new_row = {
            'subject': subject,
            'adj_length': '20s',
            'feature': 'ones',
            'train_loss': adj_ones['train_loss'][-1],
            'train_acc': adj_ones['train_acc'][-1],
            'train_auc': adj_ones['train_auc'][-1],
            'train_f1_score': adj_ones['train_f1_score'][-1],
            'val_loss': adj_ones['val_loss'][-1],
            'val_acc': adj_ones['val_acc'][-1],
            'val_auc': adj_ones['val_auc'][-1],
            'val_f1_score': adj_ones['val_f1_score'][-1],
        }
        new_row = pd.DataFrame([new_row])
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    with open(os.path.join(adj_logdir_cf, "train_val_stats.pickle"), 'rb') as file:
        adj_cf = pickle.load(file)
        new_row = {
            'subject': subject,
            'adj_length': '20s',
            'feature': 'combined',
            'train_loss': adj_cf['train_loss'][-1],
            'train_acc': adj_cf['train_acc'][-1],
            'train_auc': adj_cf['train_auc'][-1],
            'train_f1_score': adj_cf['train_f1_score'][-1],
            'val_loss': adj_cf['val_loss'][-1],
            'val_acc': adj_cf['val_acc'][-1],
            'val_auc': adj_cf['val_auc'][-1],
            'val_f1_score': adj_cf['val_f1_score'][-1],
        }
        new_row = pd.DataFrame([new_row])
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    return result_df

def calculate_stats(overall_df):
    """
       Calculates statistical tests given a DataFrame containing performance metrics for all subjects. Results are printed
       :param overall_df: DataFrame, containing performance metrics
       :return: None
    """
    overall_df = overall_df

    val_acc = overall_df[(overall_df['feature'] == 'combined') & (overall_df['adj_length'] == '1s')][
        'val_acc']
    mean_val_acc = val_acc.mean()
    val_acc_adj = overall_df[(overall_df['feature'] == 'combined') & (overall_df['adj_length'] == '20s')][
        'val_acc']
    mean_val_acc_adj = val_acc_adj.mean()
    val_loss = overall_df[(overall_df['feature'] == 'combined') & (overall_df['adj_length'] == '1s')][
        'val_loss']
    mean_val_loss = val_loss.mean()
    val_loss_adj = overall_df[(overall_df['feature'] == 'combined') & (overall_df['adj_length'] == '20s')][
        'val_loss']
    mean_val_loss_adj = val_loss_adj.mean()

    _, pvalue_acc = stats.ttest_ind(val_acc.tolist(), val_acc_adj.tolist())
    _, pvalue_loss = stats.ttest_ind(val_loss.tolist(), val_loss_adj.tolist())

    print('-' * 8 + 'Combined node and edge features' + '-' * 8)
    print(f'mean validation accuracy for 1s adj: {mean_val_acc}')
    print(f'mean validation accuracy for 20s adj: {mean_val_acc_adj}')
    print('*'*3 + f't-test on validation accuracy: p={pvalue_acc}' + '*'*3)
    print(f'mean validation loss for 1s adj: {mean_val_loss}')
    print(f'mean validation loss for 20s adj: {mean_val_loss_adj}')
    print('*' * 3 + f't-test on validation loss: p={pvalue_loss}' + '*' * 3)

    val_acc = overall_df[(overall_df['feature'] == 'ones') & (overall_df['adj_length'] == '1s')][
        'val_acc']
    mean_val_acc = val_acc.mean()
    val_acc_adj = overall_df[(overall_df['feature'] == 'ones') & (overall_df['adj_length'] == '20s')][
        'val_acc']
    mean_val_acc_adj = val_acc_adj.mean()
    val_loss = overall_df[(overall_df['feature'] == 'ones') & (overall_df['adj_length'] == '1s')][
        'val_loss']
    mean_val_loss = val_loss.mean()
    val_loss_adj = overall_df[(overall_df['feature'] == 'ones') & (overall_df['adj_length'] == '20s')][
        'val_loss']
    mean_val_loss_adj = val_loss_adj.mean()

    _, pvalue_acc = stats.ttest_ind(val_acc.tolist(), val_acc_adj.tolist())
    _, pvalue_loss = stats.ttest_ind(val_loss.tolist(), val_loss_adj.tolist())

    print('-' * 8 + 'Ones vector for node and edge features' + '-' * 8)
    print(f'mean validation accuracy for 1s adj: {mean_val_acc}')
    print(f'mean validation accuracy for 20s adj: {mean_val_acc_adj}')
    print('*'*3 + f't-test on validation accuracy: p={pvalue_acc}' + '*'*3)
    print(f'mean validation loss for 1s adj: {mean_val_loss}')
    print(f'mean validation loss for 20s adj: {mean_val_loss_adj}')
    print('*' * 3 + f't-test on validation loss: p={pvalue_loss}' + '*' * 3)

if __name__ == "__main__":

    # list of subjects to pull logs from
    subjects = ["jh101", "jh103",
                "pt01", "pt10", "pt12", "pt13", "pt14", "pt16",
                "pt2", "pt3", "pt6", "pt7", "pt8",
                "umf001",
                "ummc001", "ummc002", "ummc003", "ummc004", "ummc005", "ummc006",
                "ummc007", "ummc009"]

    overall_df = pd.DataFrame(columns=['subject', 'adj_length', 'feature', 'train_loss', 'train_acc', 'train_auc', 'train_f1_score', 'val_loss', 'val_acc', 'val_auc', 'val_f1_score'])

    for subject in subjects:
        subject_data = get_data(subject)
        overall_df = pd.concat([overall_df, subject_data], ignore_index=True)

    print(overall_df)
    overall_df.to_csv('results.csv')

    calculate_stats(overall_df)



