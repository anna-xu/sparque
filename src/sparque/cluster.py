# import nibabel as nb
from nilearn.regions import Parcellations

def run_cluster_parc(datafiles, method, n_parcels, output_name):

    cluster_parc = Parcellations(method=method, n_parcels=n_parcels,
                        standardize=False, smoothing_fwhm=2.,
                        memory='nilearn_cache', memory_level=1,
                        verbose=1)

    cluster_parc.fit(datafiles) 
    cluster_parc_img = cluster_parc.labels_img_
    cluster_parc_img.to_filename(output_name)

    return cluster_parc