import os
import pickle
from main import create_log_directories
from pathlib import Path
import pandas as pd
from scipy import stats


from scipy.stats import chisquare, ttest_ind, chi2_contingency
import numpy as np


def get_data(subject):
    """
    Reads GNN performance metric data from log directories
    :param subject: string, subject ID to pull data from
    :return: DataFrame of performance metrics
    """

    subject = subject

    # create paths for regular gres
    combined_features = True
    logdir = create_log_directories(sub=subject, node_feat="ones", edge_feat="ones", adj="coh",
                                    link_cutoff=0.3, classify="binary", combined_features=combined_features,
                                    comb_node_feat=["energy", "band_energy"], comb_edge_feat=["coh", "phase"],
                                    logdir=None, logreg=False)

    result_df = pd.read_csv(os.path.join(logdir, "GNNExplainer", "soz_freqs.csv"))
    # print(result_df)

    return result_df

def calculate_stats(overall_df):
    """
       Calculates statistical tests given a DataFrame containing performance metrics for all subjects. Results are printed
       :param overall_df: DataFrame, containing performance metrics
       :return: None
    """
    overall_df = overall_df

    # observed = overall_df[(overall_df['pred']==0) & (overall_df['truth']==0)]['seen_SOZ_nodes']
    # total = overall_df[(overall_df['pred'] == 0) & (overall_df['truth'] == 0)]['seen_nodes']
    observed = overall_df[(overall_df['pred'] == 1) & (overall_df['truth'] == 1)]['seen_SOZ_nodes']
    total = overall_df[(overall_df['pred'] == 1) & (overall_df['truth'] == 1)]['seen_nodes']
    # observed = overall_df['seen_SOZ_nodes']
    # total = overall_df['seen_nodes']

    soz_proportion = overall_df['total_SOZ_nodes'][0] / overall_df['total_nodes'][0]
    expected = [i * soz_proportion for i in total]

    observed = sum(observed)
    not_observed = sum(total) - observed
    expected = sum(expected)
    not_expected = sum(total) - expected

    print(f'observed number of SOZ nodes: {observed}')
    print(f'expected number of SOZ nodes: {expected}')
    print(f'total nodes:  {sum(total)}')

    res = chi2_contingency([[observed, not_observed], [expected, not_expected]])
    print(f'chi-square result:')
    print(res)

if __name__ == "__main__":

    # list of subjects to pull logs from
    subjects = ['ummc002', 'ummc003', 'ummc004', 'ummc005', 'ummc006', 'ummc009', 'pt01', 'pt2', 'pt3', 'pt8', 'pt13',
                'pt16', 'umf001']

    overall_df = pd.DataFrame(columns=['subject', 'seen_SOZ_nodes', 'total_SOZ_nodes', 'seen_nodes', 'total_nodes'])

    for subject in subjects:
        subject_data = get_data(subject)
        print(f'Stats for {subject}')
        calculate_stats(subject_data)
        overall_df = pd.concat([overall_df, subject_data], ignore_index=True)

    print('Stats for overall')

    calculate_stats(overall_df)



