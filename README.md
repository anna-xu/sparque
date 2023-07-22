# sparque

Software for PARcellation QUality Evaluation (sparque) is a Python package for automated evaluation of functional parcellation quality with fMRI data. **This package is currently in pre-alpha phase and <u>does not support widespread use</u>. Please see `example.py` for example of supported usage with Midnight Scan Club dataset so far. We welcome contributors.**

## Requirements & Installation
Currently, sparque works with Python 3.8+ and requires the following dependencies:
```
nibabel
nilearn
neuromaps
pandas
tables
```

To install, you can run the following lines:
```
git clone https://github.com/anna-xu/sparque
cd sparque
pip install .
```

## Package Functions
This package analyses fMRI scan data that is already BIDS-formatted. The package currently comes with 3 parcellation nifti files -- _Glasser360_, _Gordon333_, _Schaefer400 (7 network)_ -- all of which can be found in the folder __datasets__.

This package also installs majority of the DCBC package, with the surface parcellations associated with it (see [original GitHub repo](https://github.com/DiedrichsenLab/DCBC) and [paper](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.25878) from Diedrichsen Lab for more info, as well as below for info on interfacing with sparque). **To run DCBC, you will need to download a distance matrix file, which is linked on the GitHub repo.**

Outside of running parcellation quality evaluation, the sparque package also includes a few functions that may be useful:

* `run_connectivity` in the `func_conn` module returns a dataframe with the edge list of the parcelwise functional connectivity matrix for each subject and session. It outputs this dataframe in a **.csv** file and the parcellated time series in a **.h5** file. See `func_conn.py` for more information.

* `run_cluster_parc()` in `cluster.py` returns parcellation files obtained by clustering scan data (also see [nilearn clustering documentation](https://nilearn.github.io/dev/connectivity/parcellating.html) for more info)

* the `utils` module contains multiple functions that may be useful for data processing. See `utils.py` for more information. 

`run_parcel_eval()` is the main function to run sparque. It takes in a list of parcellation schemes and metrics to calculate, along with optional inputs based on the measure of interest (see `sparque.py` for more information about each input). Below contains metrics currently supported with minimal functionality:

| Metric      | Description | Required Inputs | Associated Module(s) |
| ----------- | ----------| ----------| ----------|
| Functional Connectivity Homogeneity      | Correlation of voxel timeseries within a parcel divided by number of voxels within parcel | scans,<br><br>parcellation_df | `fc_homogeneity.py` 
| Distance-Controlled Boundary Coefficient   | cluster quality metric unbiased by parcellation spatial scale| dist_file, <br><br>__data__ folder conformed to format accepted by DCBC (you can use `conform_scans_to_dcbc_dir` in the `dcbc.py` module to convert scans to fslr32k format and create data folder), <br><br>parcellation df | `dcbc.py` 
| Reliability | correlation matrix between sessions for each subject | func_conn_file (OR parcellation_df with func_conn_file column)| `reliability.py`, also see `func_conn.py` for obtaining functional connectivity files for input
| Classification accuracy | test accuracy of support vector classifier across 100 shuffled splits  | func_conn_file with `label` column (OR parcellation_df with func_conn_file column containing functional connectivity file for each parcellation)| `svc.py`, also see `func_conn.py` for obtaining functional connectivity files for input