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

def get_unique_parcels(atlas_fdata):
    unique_parcs = set(atlas_fdata.ravel().tolist()) - {0}
    return unique_parcs

# load data

def load_data(data):
    loaded_data = nb.load(data)
    fdata = loaded_data.get_fdata()

    return loaded_data, fdata 

# mask and resampling

def resample_to_data(atlas, loaded_data):
    atlas_resampled = resample_img(
                        img = atlas,
                        target_affine = loaded_data.affine,
                        target_shape = loaded_data.shape[:-1],
                        interpolation = 'nearest'
                    )
    return atlas_resampled

def get_mask(mask, data):
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
    #img_rs = img_rs.astype(np.float16)
    
    return img_rs

def filter_ts_by_std(data, data_ts_to_filter_from, std_tol_max):
   data_filtered = data[data_ts_to_filter_from.std(-1) >= std_tol_max]
   return data_filtered

# converting between spaces

def convert_mni_to_fslr32k(scan_filename, 
                           save = False,
                           filename = None):
    nb_loaded_data = nb.load(scan_filename)
    fslr_map = neuromaps.transforms.mni152_to_fslr(nb_loaded_data, '32k')

    fslr_map_L, fslr_map_R = fslr_map

    if save:
        fslr_map_L.to_filename(filename[0])
        fslr_map_R.to_filename(filename[1])
    
    return fslr_map_L, fslr_map_R

def load_multiple_scans(scans):
    loaded_scans = []
    # fdatas = []

    for _, scan in enumerate(scans):
        loaded_data, _ = load_data(scan)
        # fdatas += [scans_fdata]
        loaded_scans += [loaded_data]
    
    return loaded_scans