'''
Unit tests for reliability
'''

import numpy as np
import pandas as pd
import sparque.sparque as sparque 

def test_reliability():
    N_SUBJECTS = 2
    N_NODES = 10
    LOW_NOISE_DISPERSION = 0.03
    HIGH_NOISE_DISPERSION = 1.0

    n_edges = np.int(N_NODES * (N_NODES - 1) / 2)

    test_df = pd.DataFrame()

    for subj in range(N_SUBJECTS):
        subj_ground_truth = (np.random.rand(n_edges) - 0.5) * 2
        if subj == 0:
            ses1_noise = LOW_NOISE_DISPERSION * np.random.randn(n_edges)
            ses2_noise = LOW_NOISE_DISPERSION * np.random.randn(n_edges)
            ses1_observed = subj_ground_truth + ses1_noise
            ses2_observed = subj_ground_truth + ses2_noise
        else:
            ses1_noise = HIGH_NOISE_DISPERSION * np.random.randn(n_edges)
            ses2_noise = HIGH_NOISE_DISPERSION * np.random.randn(n_edges)
            ses1_observed = subj_ground_truth + ses1_noise
            ses2_observed = subj_ground_truth + ses2_noise
            
        ses1_observed = np.maximum(ses1_observed, -.9)
        ses1_observed = np.minimum(ses1_observed, .9)   
        ses2_observed = np.maximum(ses2_observed, -.9)
        ses2_observed = np.minimum(ses2_observed, .9)     

        subj_df = pd.DataFrame([[subj,0,*ses1_observed], [subj,1,*ses2_observed]])
        test_df = pd.concat([test_df, subj_df])
        
    test_df = test_df.rename({0:'subject', 1:'session'}, axis = 'columns')

    sub1 = sparque.run_parcel_eval(
                    ['test_parcellation'], 
                    ['reliability'],
                    parcellation_df = None,
                    func_conn_file = test_df[test_df['subject'] == 0],
                    func_conn_col_start = 2
                    )

    sub2 = sparque.run_parcel_eval(
                    ['test_parcellation'], 
                    ['reliability'],
                    parcellation_df = None,
                    func_conn_file = test_df[test_df['subject'] == 1],
                    func_conn_col_start = 2
                    )

    sub_all = sparque.run_parcel_eval(
                    ['test_parcellation'], 
                    ['reliability'],
                    parcellation_df = None,
                    func_conn_file = test_df,
                    func_conn_col_start = 2
                    )
    
    assert sub_all['reliability'].iloc[0] == np.mean([sub1['reliability'].iloc[0],sub2['reliability'].iloc[0]])

    assert sub1['reliability'].iloc[0] > sub2['reliability'].iloc[0]
