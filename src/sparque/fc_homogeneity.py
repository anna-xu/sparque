import sparque.utils as utils
import numpy as np
import pandas as pd

def calc_fc_homogeneity(atlas_fdata, fdata):
    unique_parcels = utils.get_unique_parcels(atlas_fdata)

    fc_homogeneity = []

    for _, curr_parcel in enumerate(unique_parcels):
        # if curr_parcel in null_labels:
        #     print(f'Skipping parcel {curr_parcel} because it is a null label')
        # else:
        #     print(f'Computing FC homogeneity for parcel {curr_parcel}')
        #     parcel = (atlas_fdata == curr_parcel)
        #     parcel_data = fdata[parcel.squeeze()]
        #     conn_vox_parcel = np.corrcoef(parcel_data)
        #     curr_fc_homogeneity = np.mean(conn_vox_parcel)
        #     fc_homogeneity += [curr_fc_homogeneity]
        parcel = (atlas_fdata == curr_parcel)
        parcel_data = fdata[parcel.squeeze()]
        conn_vox_parcel = np.corrcoef(parcel_data)
        curr_fc_homogeneity = np.mean(conn_vox_parcel)
        fc_homogeneity += [curr_fc_homogeneity]
    return np.mean(fc_homogeneity), fc_homogeneity 

def run_fc_homogeneity_from_dir(scans, parc_name, parc_fdata, csv_filename, surface, null_labels=()):
    fchs_df = {'parcellation': [], 'subject': [], 'session': [], 'fchs': [], 'all_fchs': []}

    for i, curr_scan in enumerate(scans):
        if surface:
            _, fdata = utils.load_data(curr_scan, is_surface=True, null_labels = null_labels)
        else:
            _, fdata = utils.load_data(curr_scan, is_surface=False, null_labels = null_labels)

        scan_split = str(curr_scan[0]).split("/")[-1].split("_")
        
        print(f'Computing functional connectivity homogeneity for scan {i}')
        
        filtered_fdata = utils.filter_ts_by_std(fdata, fdata, 1e-5)
        atlas_filtered_fdata = utils.filter_ts_by_std(parc_fdata, fdata, 1e-5)
        
        # avg_fch, subj_fch = calc_fc_homogeneity(atlas_filtered_fdata, filtered_fdata, null_labels)

        avg_fch, subj_fch = calc_fc_homogeneity(atlas_filtered_fdata, filtered_fdata)

        # for BIDS-formatted data, concatenate to subject and session
        fchs_df['parcellation'].append(parc_name)
        fchs_df['subject'].append(scan_split[0])
        fchs_df['session'].append(1)
        fchs_df['fchs'].append(avg_fch)
        fchs_df['all_fchs'].append([subj_fch])

    fchs_df = pd.DataFrame.from_dict(fchs_df)
    
    fchs_df.to_csv(csv_filename, sep=',')
    
    return fchs_df

def average_fc_homogeneity(scans, parc_name, parc_fdata, csv_filename, surface, null_labels=()):
    fchs_df = run_fc_homogeneity_from_dir(scans, parc_name, parc_fdata, csv_filename, surface, null_labels)

    avg_fc_homogeneity = fchs_df.groupby(['parcellation']).mean()

    return avg_fc_homogeneity