import numpy as np 
import os
import pandas as pd 
import nibabel as nb 
from nilearn.maskers import NiftiLabelsMasker
import sparque.utils as utils 

def subset_confounds(confounds, confounds_list, subset_confounds_dir_name):
    '''
    Subset confounds from confound files, useful if only wanting to use a few confounds from confound files.

    Parameters
    -------
    confounds : array_like
        List of filepaths as str to confound files
    confounds_list : array_like
        List of confounds as str to subset from confounds
    subset_confounds_dir_name : str
        Name of directory to output subsetted confound files
    '''
    os.mkdir(subset_confounds_dir_name)    
    for _, curr_confound_file in enumerate(confounds):
        confound_filename = str(curr_confound_file).split("/")[-1]
        curr_confounds = pd.read_csv(curr_confound_file, sep='\t')
        curr_counfounds_subset = curr_confounds[confounds_list]
        curr_counfounds_subset.to_csv(f'{subset_confounds_dir_name}/{confound_filename}', sep='\t', index = False)

def run_connectivity(parcellation_file, data, confounds = None):
    '''
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
    '''
    masker = NiftiLabelsMasker(
        labels_img=parcellation_file,
        standardize=True,
        memory='nilearn_cache',
        verbose=5,
    )

    if confounds is None:
        time_series = masker.fit_transform(data)
    else:
        time_series = masker.fit_transform(
            data,
            confounds = confounds
        )

    connectivity = np.corrcoef(time_series.T)

    return time_series, connectivity

def run_connectivity_surface(parcellation_file, data):
    '''
    Computes parcelwise connectivity matrix for surface data
    '''
    parcellation = dict(zip(('L', 'R'), parcellation_file))
    data = dict(zip(('L', 'R'), data))
    data_lab = dict()
    for hemi in ['L', 'R']:
        _,labels = utils.load_data(parcellation[hemi], is_parcellation=True, is_surface = True, null_labels = [0,-1])
        # labels = nb.load(parcellation[hemi]).darrays[0].data.astype(int)
        lab_map = np.eye(labels.max() + 1)[labels]
        data_unlab = np.stack([arr.data for arr in nb.load(data[hemi]).darrays])
        data_lab[hemi] = lab_map.T @ data_unlab.T
    # assert 0
    print('time series shape', np.vstack([data_lab['L'], data_lab['R']]).shape)
    # print('time series shape', np.corrcoef(np.vstack([data_lab['L'], data_lab['R']]).T).shape)
    return np.vstack([data_lab['L'], data_lab['R']]), np.corrcoef(np.vstack([data_lab['L'], data_lab['R']]).squeeze())

def get_uniq_conn_vals(conn_mat):
    '''
    Obtains edge list of a connectivy matrix
    '''
    return conn_mat[np.triu_indices_from(conn_mat, 1)]

def conn_from_dir(parc_name, parcellation_file, scans, confounds_subdir = None, output_name = None):   
    '''
    Run connectivity for multiple scans and saves parcellated time series in `.h5` file for each parcellation. By default, this function will create a label column with subjects as label. **NOTE: only scan names with format 'sub-{subject name}_ses-{session number}_task-rest_{run}' currently supported. 

    Parameters
    -------
    parc_name : str
        Name of parcellation
    parcellation_file : str
        Filepath to parcellation file or tuple of left and right parcellation file
    scans : array_like
        List of filepaths as str to scans or list of tuples of left and right scans
    confounds_subdir : str
        Path to associated confounds directory
    output_name (optional) : str
        If saving functional connectivity file, return 
    
    Returns
    -------
    conn_df : dataframe 
        Dataframe containing edge list of parcelwise connectivty matrix for each subject and session
    ''' 
    conn_df = pd.DataFrame()
    subjects = []
    sessions = []

    time_series_df = {'subject': [], 'session': [], 'time_series': []}

    for curr_scan in scans:
        if isinstance(curr_scan, tuple):
            # TO DO LATER: incorporate session info
            subj_ses_info = curr_scan[0].split("/")[-1].split("_")
            subjects += [subj_ses_info[0]]
            sessions += [1]
            print(f'Currently computing connectivity based on surface data for {subj_ses_info[0]}')
            time_series_df['subject'].append(subj_ses_info[0])
            time_series_df['session'].append(1)
            curr_time_series, curr_conn_mat = run_connectivity_surface(parcellation_file, curr_scan)
        else:
            scan_split = str(curr_scan).split("/")[-1].split("_")

            if confounds_subdir is not None:
                confound_file = f'{confounds_subdir}/{scan_split[0]}_{scan_split[1]}_task-rest_{scan_split[3]}_desc-confounds_timeseries.tsv'
            else:
                confound_file = None 

            print(f'Currently computing for {scan_split[0]}, {scan_split[1]} with confound file {confound_file}')

            subjects += [scan_split[0]]
            sessions += [scan_split[1]]

            time_series_df['subject'].append(scan_split[0])
            time_series_df['session'].append(scan_split[1])

            curr_time_series, curr_conn_mat = run_connectivity(parcellation_file, curr_scan, confound_file)

        time_series_df['time_series'].append(curr_time_series)
        
        curr_conn_uq = get_uniq_conn_vals(curr_conn_mat)
        subj_ses = np.array([subjects[-1], sessions[-1]])
        # subj_ses = np.array([scan_split[0], scan_split[1]])

        curr_data = np.concatenate((subj_ses, curr_conn_uq))
        curr_data_df = pd.DataFrame(data = curr_data)

        curr_data_df = curr_data_df.T

        conn_df = pd.concat([conn_df, curr_data_df], axis = 0)

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