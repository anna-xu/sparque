import os

import pandas as pd

import sparque.utils as utils

from .DCBC import eval_DCBC, plotting


def compute_DCBC(nb_loaded_parcel_gii, hem, dist_file, plot=False):
    """
    Function used in `run_DCBC()` to run DCBC per scan and hemisphere.
    """
    parcels = nb_loaded_parcel_gii.darrays[0].data

    myDCBC = eval_DCBC.DCBC(
        hems=hem,
        maxDist=35,
        binWidth=2.5,
        dist_file=dist_file,
    )

    T = myDCBC.evaluate(parcels)

    if plot:
        plotting.plot_wb_curve(T, path='data', hems=hem)

    return T


def conform_scans_to_dcbc_dir(scans):
    """
    Conforms scans to accepted DCBC inputs by saving a `data` folder with scans projected
    onto fslr32k space for DCBC analysis.

    .. warning::
        only tested to run for scan file names with ``sub-{sub_name}_ses-{session_name}``
        format, which assumes 1 nifti file per session
    """
    if os.path.isdir('data') is False:
        os.mkdir('data')

    for curr_scan in scans:
        scan_split = str(curr_scan).split('/')[-1].split('_')

        subdir_path = os.path.join('data', f'{scan_split[0]}_{scan_split[1]}')

        if os.path.exists(subdir_path):
            continue
        else:
            os.makedirs(subdir_path)

            utils.convert_mni_to_fslr32k(
                curr_scan,
                save=True,
                filename=[
                    f'data/{scan_split[0]}_{scan_split[1]}/{scan_split[0]}'
                    f'_{scan_split[1]}.L.wbeta.32k.func.gii',
                    f'data/{scan_split[0]}_{scan_split[1]}/{scan_split[0]}'
                    f'_{scan_split[1]}.R.wbeta.32k.func.gii',
                ],
            )


def run_DCBC(dist_file, parc_name, parcel_filenames, csv_filename=None):
    """
    Function used by ``run_all_metrics()`` to run DCBC. Can be used without ``sparque``
    wrapper.

    Parameters:
    -------
    dist_file : str
        Filepath to distance matrix file
    parc_name : str
        Name of parcellation to run
    parcel_filenames : array_like
        List of filepaths as str to left surface image of parcellation and right surface
        image (please make sure order is right)
    csv_filename (optional) : str
        Name of output to save if desired. Must end in ``.csv``

    Returns:
    -------
    DCBC_df : dataframe
        full dataframe of output from DCBC function
    DCBC_average : dataframe
        minimal dataframe containing only hemisphere and DCBC value

    """
    parcel_fslr_map_L = parcel_filenames[0]
    parcel_fslr_map_R = parcel_filenames[1]

    L_myDCBC = compute_DCBC(parcel_fslr_map_L, 'L', dist_file)
    L_myDCBC = pd.DataFrame.from_dict(L_myDCBC)
    L_myDCBC = L_myDCBC.T
    L_myDCBC['parcellation'] = parc_name

    R_myDCBC = compute_DCBC(parcel_fslr_map_R, 'R', dist_file)
    R_myDCBC = pd.DataFrame.from_dict(R_myDCBC)
    R_myDCBC = R_myDCBC.T
    R_myDCBC['parcellation'] = parc_name

    DCBC_df = pd.concat([L_myDCBC, R_myDCBC])

    DCBC_df['DCBC'] = DCBC_df['DCBC'].astype('float64')

    DCBC_df.to_csv(csv_filename, sep=',')

    DCBC_average = DCBC_df[['hemisphere', 'DCBC']]

    return DCBC_df, DCBC_average
