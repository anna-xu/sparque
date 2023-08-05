import os

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker


def subset_confounds(confounds, confounds_list, subset_confounds_dir_name):
    """
    Subset confounds from confound files, useful if only wanting to use a few confounds
    from confound files.

    Parameters
    -------
    confounds : array_like
        List of filepaths as str to confound files
    confounds_list : array_like
        List of confounds as str to subset from confounds
    subset_confounds_dir_name : str
        Name of directory to output subsetted confound files
    """
    os.mkdir(subset_confounds_dir_name)
    for curr_confound_file in confounds:
        confound_filename = str(curr_confound_file).split('/')[-1]
        curr_confounds = pd.read_csv(curr_confound_file, sep='\t')
        curr_counfounds_subset = curr_confounds[confounds_list]
        curr_counfounds_subset.to_csv(
            f'{subset_confounds_dir_name}/{confound_filename}',
            sep='\t',
            index=False,
        )


def run_connectivity(parcellation_file, data, confounds):
    """
    Computes parcelwise connectivity matrix

    Parameters
    -------
    parcellation_file : str
        Filepath to parcellation file
    data : str
        Filepath to image to compute parcelwise connectiviy matrix
    confounds : str
        Filepath to associated confound file

    Returns
    -------
    time_series : array_like
        parcellated time series
    connectivity : array_like
        parcelwise connectivy matrix
    """
    masker = NiftiLabelsMasker(
        labels_img=parcellation_file,
        standardize=True,
        memory='nilearn_cache',
        verbose=5,
    )

    time_series = masker.fit_transform(data, confounds=confounds)

    connectivity = np.corrcoef(time_series.T)

    return time_series, connectivity


def get_uniq_conn_vals(conn_mat):
    """
    Obtains edge list of a connectivy matrix
    """
    return conn_mat[np.triu_indices_from(conn_mat, 1)]


def conn_from_dir(
    parc_name,
    parcellation_file,
    scans,
    confounds_subdir,
    output_name=None,
):
    """
    Run connectivity for multiple scans and saves parcellated time series in `.h5` file
    for each parcellation. By default, this function will create a label column with
    subjects as label.

    .. note::
        only scan names with format
        ``'sub-{subject name}_ses-{session number}_task-rest_{run}'`` currently
        supported.

    Parameters
    -------
    parc_name : str
        Name of parcellation
    parcellation_file : str
        Filepath to parcellation file
    scans : array_like
        List of filepaths as str to scans
    confounds_subdir : str
        Path to associated confounds directory
    output_name (optional) : str
        If saving functional connectivity file, return

    Returns
    -------
    conn_df : dataframe
        Dataframe containing edge list of parcelwise connectivty matrix for each subject
        and session
    """
    conn_df = pd.DataFrame()
    subjects = []
    sessions = []

    time_series_df = {'subject': [], 'session': [], 'time_series': []}

    for curr_scan in scans:
        scan_split = str(curr_scan).split('/')[-1].split('_')

        confound_file = (
            f'{confounds_subdir}/{scan_split[0]}_{scan_split[1]}_task-rest_'
            f'{scan_split[3]}_desc-confounds_timeseries.tsv'
        )

        print(
            f'Currently computing for {scan_split[0]}, {scan_split[1]} with confound '
            f'file {confound_file}'
        )

        subjects += [scan_split[0]]

        sessions += [scan_split[1]]

        time_series_df['subject'].append(scan_split[0])
        time_series_df['session'].append(scan_split[1])

        curr_time_series, curr_conn_mat = run_connectivity(
            parcellation_file, curr_scan, confound_file
        )

        time_series_df['time_series'].append(curr_time_series)

        curr_conn_uq = get_uniq_conn_vals(curr_conn_mat)
        subj_ses = np.array([scan_split[0], scan_split[1]])

        curr_data = np.concatenate((subj_ses, curr_conn_uq))
        curr_data_df = pd.DataFrame(data=curr_data)

        curr_data_df = curr_data_df.T

        conn_df = pd.concat([conn_df, curr_data_df], axis=0)

    conn_df.rename(columns={0: 'subject', 1: 'session'}, inplace=True)
    conn_df['label'] = conn_df['subject']

    if output_name is None:
        print('functional connectivity file not exported')
    else:
        conn_df.to_csv(output_name, sep=',')

    time_series_df = pd.DataFrame.from_dict(time_series_df)
    time_series_store = pd.HDFStore(f'{parc_name}_time_series.h5')
    time_series_store['df'] = time_series_df
    time_series_store.close()

    return conn_df
