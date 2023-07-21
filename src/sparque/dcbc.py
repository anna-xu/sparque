from pathlib import Path 
import os 
import pandas as pd

import sparque.utils as utils
from .DCBC import eval_DCBC
from .DCBC import plotting

def compute_DCBC(nb_loaded_parcel_gii, hem, dist_file, plot=False):
    parcels = nb_loaded_parcel_gii.darrays[0].data

    myDCBC = eval_DCBC.DCBC(hems = hem, maxDist = 35, binWidth = 2.5, dist_file = dist_file)

    T = myDCBC.evaluate(parcels)

    if plot:
        plotting.plot_wb_curve(T, path = 'data', hems = hem)

    return T

def conform_scans_to_dcbc_dir(scans):
    if os.path.isdir('data') == False:
        os.mkdir('data')

    for _, curr_scan in enumerate(scans):
        scan_split = str(curr_scan).split("/")[-1].split("_")

        subdir_path = os.path.join('data', f'{scan_split[0]}_{scan_split[1]}')
        
        if os.path.exists(subdir_path):
            continue
        else:
            os.makedirs(subdir_path)
            
            _, _ = utils.convert_mni_to_fslr32k(curr_scan, save=True, filename = [f'data/{scan_split[0]}_{scan_split[1]}/{scan_split[0]}_{scan_split[1]}.L.wbeta.32k.func.gii', f'data/{scan_split[0]}_{scan_split[1]}/{scan_split[0]}_{scan_split[1]}.R.wbeta.32k.func.gii'])

def run_DCBC(dist_file,
             parc_name,
             parcel_filename,
             run_mni_to_fslr32k = True, 
             scan_directory = None,
             convert_parcel_to_fslr32k = True,
             save_parcel_gii = False, 
             filename_gii = None,
             csv_filename = None):
    
    if run_mni_to_fslr32k:
        scans = list(Path(scan_directory).glob('*.nii.gz'))
        for i, curr_scan in enumerate(scans):
            scan_split = str(curr_scan).split("/")[1].split("_")

            subdir_path = os.path.join('data', f'{scan_split[0]}_{scan_split[1]}')
            
            os.makedirs(subdir_path)
            
            fslr_map_L, fslr_map_R = utils.convert_mni_to_fslr32k(curr_scan, save=True, filename = [f'data/{scan_split[0]}_{scan_split[1]}/{scan_split[0]}_{scan_split[1]}.L.wbeta.32k.func.gii', f'data/{scan_split[0]}_{scan_split[1]}/{scan_split[0]}_{scan_split[1]}.R.wbeta.32k.func.gii'])
    
    if convert_parcel_to_fslr32k:
        parcel_fslr_map_L, parcel_fslr_map_R = utils.convert_mni_to_fslr32k(parcel_filename, 
                                                                      save = save_parcel_gii,
                                                                      filename = filename_gii
                                                                     )
    else:
        # parcel_fslr_map_L = nb.load(parcel_filename[0])
        # parcel_fslr_map_R = nb.load(parcel_filename[1])
        parcel_fslr_map_L = parcel_filename[0]
        parcel_fslr_map_R = parcel_filename[1]

    L_myDCBC = compute_DCBC(parcel_fslr_map_L, 'L', dist_file)
    L_myDCBC = pd.DataFrame.from_dict(L_myDCBC)
    L_myDCBC = L_myDCBC.T
    L_myDCBC['parcellation'] = parc_name
    # L_myDCBC = pd.DataFrame.from_dict(L_myDCBC)

    R_myDCBC = compute_DCBC(parcel_fslr_map_R, 'R', dist_file)
    R_myDCBC = pd.DataFrame.from_dict(R_myDCBC)
    R_myDCBC = R_myDCBC.T
    R_myDCBC['parcellation'] = parc_name
    # R_myDCBC = pd.DataFrame.from_dict(R_myDCBC)

    DCBC_df = pd.concat([L_myDCBC, R_myDCBC])

    print(DCBC_df)

    DCBC_df['DCBC'] = DCBC_df['DCBC'].astype('float64')

    DCBC_df.to_csv(csv_filename, sep = ',')

    DCBC_average = DCBC_df[['hemisphere', 'DCBC']]
    # DCBC_average = DCBC_df.group_by(['hemisphere']).mean()

    return DCBC_df, DCBC_average