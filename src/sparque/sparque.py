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
                    func_conn_col_start = 3):
    '''
    Inputs:
    scans - either a directory of scans or a list of nibabel loaded scans
    parc_name - name of parcellation as a string
    parc_fdata - optional argument, float data for parcellation for calculating functional homogeneity
    loaded_surface_parc - optional argument, list of the parcellation (filename as string) loaded as surface data
    func_conn_file - optional argument to use with reliability or svc; a file with the upper triangle of the functional connectivity matrix for each scan OR a dataframe
    num_iteractions - optional argument to include number of iterations to run for reliability
    frac_to_sample - optional argument to include fraction of data to sample during each iteraction of reliability estimate
    '''

    eval_data = {'parcellation': [parc_name]}

    for _, curr_metric in enumerate(metrics):
        print(f'Computing {curr_metric}')
        if curr_metric == 'fc_homogeneity':
            temp_eval_data_df = fc_homogeneity.average_fc_homogeneity(scans, parc_name, parc_fdata, f'{parc_name}_fch_{datetime.now()}.csv')

            eval_data['fc_homogeneity'] = [temp_eval_data_df['fchs'].iloc[0]]
        
        elif curr_metric == 'reliability':

            # print(reliability.run_avg_corr_connmats)

            temp_avg_corr_connmat_df = reliability.reliability_multiple_subjects(reliability_conn_file, 'subject', f'{parc_name}_reliabilities_{datetime.now()}.h5', func_conn_col_start)

            temp_eval_data_df = reliability.get_reliability(temp_avg_corr_connmat_df)

            # print(temp_eval_data_df['avg_corr'].iloc[0])
            eval_data['reliability'] = np.mean(temp_eval_data_df['mean_reliability'])
            # eval_data['reliability'] = [temp_eval_data_df['reliability'].iloc[0]]
            
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

        # eval_data_df = pd.merge(eval_data_df, temp_eval_data_df, on = ['parcellation'])

    eval_data_df = pd.DataFrame.from_dict(eval_data)
    
    return eval_data_df

def run_parcel_eval(parcellations, 
                    metrics, 
                    scans = None,
                    parcellation_df = parcellation_dict.parcellation_df,
                    dist_file = None,
                    func_conn_file = None,
                    func_conn_col_start = 3):

    '''
    n_parcels (optional): a list of number of parcels for each parcellation, include None in list if number of parcels is not relevant

    dist_file (optional): location of distance matrix file for DCBC

    func_conn_file (optional): csv file containing subject, session, and upper triangle of functional connectivity matrix; to be used to measure reliability and classification accuracy
    '''
    metric_dfs = []

    # run cluster parcellations
    for _, curr_parc in enumerate(parcellations):
        print(f'Computing {curr_parc}') 

        if parcellation_df is not None:
            parcel_surface_L_data = parcellation_df['surface_file_L'][parcellation_df['parcellation'] == curr_parc].iloc[0]

            parcel_surface_R_data = parcellation_df['surface_file_R'][parcellation_df['parcellation'] == curr_parc].iloc[0]

            func_conn_file = parcellation_df['func_conn_file'][parcellation_df['parcellation'] == curr_parc].iloc[0]

        if 'fc_homogeneity' in metrics:
            parcellation_file = parcellation_df['parc_file'][parcellation_df['parcellation'] == curr_parc].iloc[0]

            _, parc_fdata = utils.load_data(parcellation_file)
        else:
            parc_fdata = None
            
        if 'reliability' in metrics: 
            if isinstance(func_conn_file, pd.DataFrame):
                conn_df = func_conn_file
            else:
                conn_df = pd.read_csv(func_conn_file)
            reliability_df = conn_df.copy()
            if 'label' in reliability_df.columns:
                reliability_df = reliability_df.drop('label', axis = 'columns')
            # reliability_df = reliability.column_to_list('fc', fc_column_start, reliability_df)
            # reliability_df.to_csv(f'{curr_parc}_reliability_df_used.csv')
        else:
            reliability_df = None
            conn_df = func_conn_file 

        if 'dcbc' in metrics:
            parc_L_gii, parc_R_gii = nb.load(parcel_surface_L_data), nb.load(parcel_surface_R_data)
            surface_parc = [parc_L_gii, parc_R_gii]
        else:
            surface_parc = None 

        temp_eval_data = run_all_metrics(scans, metrics, curr_parc, parc_fdata, dist_file, surface_parc, conn_df, reliability_df, func_conn_col_start)

        metric_dfs += [temp_eval_data]
        
    parc_metric_df = pd.concat(metric_dfs)

    parc_metric_df.to_csv(f'parcellation_metrics_{datetime.now()}.csv')

    return parc_metric_df