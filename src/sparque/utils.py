import numpy as np
import pandas as pd
import nibabel as nb
import neuromaps
from nilearn.image import resample_img
import importlib 

def sem(array):
    return np.std(array) / np.sqrt(np.size(array))

def dict_to_csv(dict, csv_filename):
    df = pd.DataFrame(data = dict)
    df.to_csv(csv_filename, sep = ',')
    return df

def convert_mni_to_fslr32k(scan_filename, 
                           save = False,
                           filename = None):
    '''
    Wrapper function to project scans onto fslr32k space

    Parameters
    -----
    scan_filename : str
        Filepath of scan
    save (optional): bool
        If true, saves the scan
    filename (optional): array_like
        List of left surface filename and right surface filename

    Returns
    -----
    fslr_map_L : nibabel loaded object of left fslr map
    fslr_map_R : nibabel loaded object of right fslr map
    '''
    nb_loaded_data = nb.load(scan_filename)
    fslr_map = neuromaps.transforms.mni152_to_fslr(nb_loaded_data, '32k')

    fslr_map_L, fslr_map_R = fslr_map

    if save:
        fslr_map_L.to_filename(filename[0])
        fslr_map_R.to_filename(filename[1])
    
    return fslr_map_L, fslr_map_R


def get_unique_parcels(atlas_fdata):
    unique_parcs = set(atlas_fdata.ravel().tolist()) - {0}
    return unique_parcs

def load_data(data, is_parcellation = False, is_surface = False, null_labels=()):
    '''
    Loads scan data via nibabel as outputs the loaded scan and fdata
    '''
    loaded_data = nb.load(data)

    if is_surface:
        print('Loading surface data')
        fdata = np.stack([arr.data for arr in loaded_data.darrays]).T
    else:
        fdata = loaded_data.get_fdata()

    if is_parcellation:
        print('Loading parcellation data')
        fdata = fdata.astype(int)
        for label in null_labels:
            if label < 0:
                label = fdata.max() + 1 + label 
            fdata[fdata == label] = 0

    return loaded_data, fdata 

def resample_to_data(atlas, loaded_data):
    '''
    Resamples parcellation file (atlas) to nibabel loaded scan data (loaded_data) 
    '''
    atlas_resampled = resample_img(
                        img = atlas,
                        target_affine = loaded_data.affine,
                        target_shape = loaded_data.shape[:-1],
                        interpolation = 'nearest'
                    )
    return atlas_resampled

def get_mask(mask, data):
    '''
    Resamples to mask
    '''
    if (mask == 'MNI' and importlib.util.find_spec('templateflow')):
        import templateflow.api as tflow 
        img_mask = tflow.get('MNI152NLin2009cAsym', desc='brain', suffix='mask', resolution=2)
    else: 
        img_mask = mask

    mask_for_rs = resample_img(
        img = img_mask,
        target_affine = data.affine,
        target_shape = data.shape[:-1],
        interpolation = 'nearest'
    )

    return mask_for_rs

def mask_data(mask_for_rs, data):
    data = data.get_fdata()
    img_rs = data[mask_for_rs.get_fdata().astype(np.bool)]
    return img_rs

def filter_ts_by_std(data, data_ts_to_filter_from, std_tol_max):
    '''
    Filters out voxels with time series containing standard deviation of less than std_tol_max
    '''
    data_filtered = data[data_ts_to_filter_from.std(-1) >= std_tol_max]
    return data_filtered

def load_multiple_scans(scans):
    loaded_scans = []

    for _, scan in enumerate(scans):
        loaded_data, _ = load_data(scan)
        loaded_scans += [loaded_data]
    
    return loaded_scans