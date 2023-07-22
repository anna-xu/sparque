# Example workflow running sparque with Midnight Scan Club dataset
from sparque import utils
from sparque import parcellation_dict
from sparque import func_conn
from sparque import cluster
from sparque import sparque

import glob

# Keep this since parcellation_dict assumes the existence of these parcellations
parcellations = ['schaefer2018', 'gordon2016', 'glasser2016']

# Specify scans to run sparque on
scans = glob.glob('ds000224-fmriprep/*/*/func/*.nii.gz', recursive=True)

# Load parcellation dictionary dataframe
parcellation_df = parcellation_dict.parcellation_df

# Run ward parcellation
loaded_scans = utils.load_multiple_scans(scans)
ward_parcellation_file = f'ward400_parcellation.nii.gz'
_, parc_fdata = cluster.run_cluster_parc(loaded_scans, 'ward', 400, ward_parcellation_file)

ward_surface_R = 'ward400_surface_R.nii.gz'
ward_surface_L = 'ward400_surface_L.nii.gz'
utils.convert_mni_to_fslr32k(ward_parcellation_file, True, [ward_surface_L, ward_surface_R])

# add ward to parcellation dictionary
ward_info = {'parcellation': 'ward',
                'parc_file': ward_parcellation_file,
                'n_parcels': 400,
                'surface_file_R': ward_surface_R,
                'surface_file_L': ward_surface_L}

parcellation_df = parcellation_df.append(ward_info, ignore_index = True) 

# Run connectivity for each parcellation
confounds = glob.glob('ds000224-fmriprep/*/*/func/*.tsv', recursive=True)

func_conn.subset_confounds(confounds, ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter', 'global_signal'], 'subset_confounds')

subsetted_confounds = glob.glob(f'subset_confounds/*.tsv', recursive=True)

for _, curr_parc in enumerate(parcellations):
    conn_df = func_conn.conn_from_dir(curr_parc, parcellation_df['parc_file'][parcellation_df['parcellation'] == curr_parc].iloc[0], scans, subsetted_confounds, f'{curr_parc}_func_conn.csv')

# Example code if want to change label to something other than subject
# schaefer_fc = pd.read_csv('schaefer2018_func_conn.csv')
# schaefer_fc['label'] = ...
# schaefer_fc.to_csv('schaefer2018_func_conn.csv', ignore_index = True)

# Add functional connectivity file to dictionary
# Note: this works even if you didn't run functional connectivity for all the parcellations
connectivity_files = []
for _, curr_parc in enumerate(parcellations):
    connectivity_files.append(f'{curr_parc}_func_conn.csv')

parcellation_df['func_conn_file'] = connectivity_files

# Run sparque
sparque.run_parcel_eval(parcellations = parcellations + ['ward'], 
                    metrics = ['fc_homogeneity', 'dcbc', 'reliability', 'svc'], 
                    scans = scans,
                    parcellation_df = parcellation_df,
                    dist_file = 'distanceMatrix/distSphere_sp.mat',
                    func_conn_file = None,
                    func_conn_col_start = 3)
