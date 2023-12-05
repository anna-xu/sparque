import pandas as pd
from datetime import datetime
import nibabel as nb
import numpy as np

import sparque.fc_homogeneity as fc_homogeneity
import sparque.utils as utils
import sparque.reliability as reliability
import sparque.svc as svc
import sparque.dcbc as dcbc
import sparque.parcellation_dict as parcellation_dict

def run_all_metrics(scans,  
                    metrics,
                    parc_name, 
                    parc_fdata = None, 
                    dist_file = None,
                    loaded_surface_parc = None, 
                    func_conn_file = None,
                    reliability_conn_file = None,
                    func_conn_col_start = 3,
                    surface=False,
                    null_labels = ()):
    '''
    Function to save metric outputs (used in `run_parcel_eval()`)
    '''

    eval_data = {'parcellation': [parc_name]}

    for _, curr_metric in enumerate(metrics):
        print(f'Computing {curr_metric}')
        if curr_metric == 'fc_homogeneity':
            temp_eval_data_df = fc_homogeneity.average_fc_homogeneity(scans, parc_name, parc_fdata, f'{parc_name}_fch_{datetime.now()}.csv', surface, null_labels)

            eval_data['fc_homogeneity'] = [temp_eval_data_df['fchs'].iloc[0]]
        
        elif curr_metric == 'reliability':

            temp_avg_corr_connmat_df = reliability.reliability_multiple_subjects(reliability_conn_file, 'subject', f'{parc_name}_reliabilities_{datetime.now()}.h5', func_conn_col_start)

            temp_eval_data_df = reliability.get_reliability(temp_avg_corr_connmat_df)

            eval_data['reliability'] = np.mean(temp_eval_data_df['mean_reliability'])
            
        elif curr_metric == 'svc':
            mean_acc, scores = svc.run_svc_with_shuffle_split(func_conn_file)
            temp_eval_dict = {'parcellation': [parc_name], 'svc': [scores], 'svc_mean_acc': [mean_acc]}
            temp_eval_data_df = pd.DataFrame.from_dict(temp_eval_dict)

            svc_store = pd.HDFStore(f'{parc_name}_svc_{datetime.now()}.h5')
            svc_store['df'] = temp_eval_data_df
            svc_store.close()

            eval_data['svc'] = [temp_eval_data_df['svc_mean_acc'].iloc[0]]

        elif curr_metric == 'dcbc':
            _, DCBC_average_df = dcbc.run_DCBC(dist_file,
                                              parc_name,
                                              loaded_surface_parc,
                                              run_mni_to_fslr32k = False, 
                                              scan_directory = None,
                                              convert_parcel_to_fslr32k = False,
                                              save_parcel_gii = False, 
                                              filename_gii = None,
                                              csv_filename = f'{parc_name}_DCBC_{datetime.now()}.csv')
            temp_eval_data_df = DCBC_average_df

            eval_data['L_DCBC'] = [temp_eval_data_df['DCBC'][temp_eval_data_df['hemisphere'] == 'L']]
            eval_data['R_DCBC'] = [temp_eval_data_df['DCBC'][temp_eval_data_df['hemisphere'] == 'R']]

    eval_data_df = pd.DataFrame.from_dict(eval_data)
    
    return eval_data_df

def run_parcel_eval(parcellations, 
                    metrics, 
                    scans = None,
                    surface = False,
                    null_labels = (),
                    parcellation_df = parcellation_dict.parcellation_df,
                    dist_file = None,
                    func_conn_file = None,
                    func_conn_col_start = 3):
    """
    Wrapper function to run specified parcellations and metrics. 

    Parameters
    ----------
    parcellations : array_like
        List of parcellation names as str to run 
    metrics : array_like
        List of metrics as str to run (currently only accepts 'fc_homogeneity', 'dcbc', 'reliability', 'svc)
    scans (optional): array_like
        List of scans to analyze 
    parcellation_df : dataframe object
        Dataframe containing parcellation name, associated parcellation file, number of parcels, associated surface image file (left), associated surface image file (right). You can run `parcellation_dict` and look at `parcellation_df` for example of default. 
    dist_file (optional): str
        Location of distance matrix file for DCBC. Please see DCBC GitHub repo for more information on obtaining distance matrix file (https://github.com/DiedrichsenLab/DCBC). Since this file is big, it cannot be readily uploaded onto GitHub repo.
    func_conn_file (optional): str or Dataframe 
        csv file containing subject, session, and upper triangle of functional connectivity matrix; to be used to measure reliability and classification accuracy
    func_conn_col_start : int
        Index of where the edge list values start. If output from `func_conn.conn_from_dir`, edge list values start at column 3. 

    Returns
    -------
    array_like  
        Upper triangle of matrix of reliabilities across pairs of runs 
    """
    metric_dfs = []

    for _, curr_parc in enumerate(parcellations):
        print(f'Computing {curr_parc}') 

        if parcellation_df is not None:
            parcel_surface_L_data = parcellation_df['surface_file_L'][parcellation_df['parcellation'] == curr_parc].iloc[0]
            parcel_surface_R_data = parcellation_df['surface_file_R'][parcellation_df['parcellation'] == curr_parc].iloc[0]
            

        if 'fc_homogeneity' in metrics:
            parcellation_file = parcellation_df['parc_file'][parcellation_df['parcellation'] == curr_parc].iloc[0]
            _, parc_fdata = utils.load_data(parcellation_file, is_parcellation = True, is_surface = surface, null_labels=null_labels)
            func_conn_file = None
        else:
            func_conn_file = parcellation_df['func_conn_file'][parcellation_df['parcellation'] == curr_parc].iloc[0]
            parc_fdata = None

        if func_conn_file is not None:
            if isinstance(func_conn_file, pd.DataFrame):
                conn_df = func_conn_file
            else:
                conn_df = pd.read_csv(func_conn_file)

        if 'reliability' in metrics: 
            reliability_df = conn_df.copy()
            if 'label' in reliability_df.columns:
                reliability_df = reliability_df.drop('label', axis = 'columns')
        else:
            reliability_df = None
            conn_df = func_conn_file 

        if 'dcbc' in metrics:
            parc_L_gii, parc_R_gii = nb.load(parcel_surface_L_data), nb.load(parcel_surface_R_data)
            surface_parc = [parc_L_gii, parc_R_gii]
        else:
            surface_parc = None 

        temp_eval_data = run_all_metrics(scans, metrics, curr_parc, parc_fdata, dist_file, surface_parc, conn_df, reliability_df, func_conn_col_start, surface, null_labels)

        metric_dfs += [temp_eval_data]
        
    parc_metric_df = pd.concat(metric_dfs)

    parc_metric_df.to_csv(f'parcellation_metrics_{datetime.now()}.csv')

    return parc_metric_df