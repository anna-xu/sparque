import pandas as pd
import numpy as np
from datetime import datetime

# fisher z-transform
def z_transform_corr(correlations):
    z_corrs = []
    for i,corr in enumerate(correlations):
        z_corrs += [np.arctanh(corr)]
    return z_corrs

def calc_reliability(subj_name, df, subj_column_name, func_conn_col_start=3):
    """
    Calculate reliability of parcellated connectome across runs for a single subject.

    We define reliability as the correlation coefficient between the (Fisher z-transformed) edge lists of the connectivity matrices. If there are more than 2 runs, the output will be the list of unique edge values of a correlation coefficient matrix, with each value of the matrix representing the reliability between two runs.

    Parameters
    ----------
    subj_name : str
        Name of subject to calculate reliability for.
    df : dataframe object
        Dataframe of edge lists of connectivity matrices across runs with subject and run columns specified (currently expected output from `func_conn.conn_from_dir`)
    subj_column_name : str
        Column name that defines the subject
    func_conn_col_start : int
        Index of where the edge list values start. If output from `func_conn.conn_from_dir`, edge list values start at column 3. 

    Returns
    -------
    array_like  
        Upper triangle of matrix of reliabilities across pairs of runs 
    """

    # cut the dataframe for only one subject
    test_subj = df[df[subj_column_name] == subj_name]
    func_conn_mat = test_subj.iloc[:,func_conn_col_start:].values
    z_func_conn_mat = np.arctanh(func_conn_mat)
    corr_conn = np.corrcoef(z_func_conn_mat)
    if corr_conn.shape == ():
        print(test_subj.shape)
        print(test_subj[subj_column_name])

    return corr_conn[np.triu_indices_from(corr_conn, 1)]

def reliability_multiple_subjects(df, subj_column_name, csv_filename, func_conn_col_start=3):
    nan_df = df.isna().any(axis=1)

    with open(f'reliability_log_{datetime.now()}.txt', 'w') as f:
        f.write(f'rows dropped \n {df[nan_df]}')

    df = df.dropna(axis = 'rows')
    
    # this function for multiple subjects
    subjects = df[subj_column_name]

    avg_corr_connmats = {'subject': [], 'reliabilities': []}

    for _,subject in enumerate(subjects):
        reliabilities = calc_reliability(subject, df, subj_column_name, func_conn_col_start)
        avg_corr_connmats['subject'].append(subject)
        avg_corr_connmats['reliabilities'].append(reliabilities)

    avg_corr_connmats_df = pd.DataFrame.from_dict(avg_corr_connmats)

    avg_corr_conmats_store = pd.HDFStore(csv_filename)
    avg_corr_conmats_store['df'] = avg_corr_connmats_df
    avg_corr_conmats_store.close()

    # avg_corr_connmats_df.to_csv(csv_filename, sep = ',')

    return avg_corr_connmats_df

def get_reliability(avg_corr_connmats_df):
    reliability_df = {'subject': [], 'mean_reliability': []}
    for i,subject in enumerate(avg_corr_connmats_df['subject']):
        reliability_df['subject'].append(subject)
        reliability_df['mean_reliability'].append(np.mean(avg_corr_connmats_df['reliabilities'].iloc[i]))
    # reliability_df = avg_corr_connmats_df.groupby(['subject']).mean()
    return reliability_df
