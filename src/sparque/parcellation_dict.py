import pandas as pd 

# parcellation names
parcellations = ['schaefer2018', 'gordon2016', 'glasser2016']

# file names of parcellations

parc_file = ['datasets/volume_parcellations/schaefer400_resampled_myconnectome.nii.gz', 'datasets/volume_parcellations/gordon333_resampled_myconnectome.nii.gz', 'datasets/volume_parcellations/glasser360_resampled_myconnectome.nii.gz']

# file names of parcellations in surface space
surface_files_L = ['DCBC/parcellations/Schaefer2018_7Networks_400.32k.L.label.gii', 'DCBC/parcellations/Gordon.32k.L.label.gii', 'DCBC/parcellations/Glasser_2016.32k.L.label.gii']

surface_files_R = ['DCBC/parcellations/Schaefer2018_7Networks_400.32k.R.label.gii', 'DCBC/parcellations/Gordon.32k.R.label.gii', 'DCBC/parcellations/Glasser_2016.32k.R.label.gii']

# number of parcels for each parcellation
n_parcels = [400, 333, 360]

# parcellation dictionary
parcellation_dict = {'parcellation': parcellations,
                'parc_file': parc_file,
                'n_parcels': n_parcels,
                'surface_file_R': surface_files_R,
                'surface_file_L': surface_files_L}

parcellation_df = pd.DataFrame.from_dict(parcellation_dict)