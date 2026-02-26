calculate_photon_weighted_PCCs
Purpose
Computes photon-weighted Pearson correlation coefficients between LptD and newly inserted LPS for each cell, including pole and mid-cell regions.

Inputs
One CSV table: demograph_combined_spreadsheet_photons.csv with Cell_ID (or CellID), Channel, PositionX_nm_, PositionY_nm_, and Photons (also accepts Photons_ or PhotonCount).

Parameters
binWidth_nm = 25 nm and sigma_bins = 1.5 bins are fixed to match the Methods.

Outputs
Writes PhotonWeighted_PCC_Output.csv with PCCs for Pole1, Mid, Pole2 and Whole cell per Cell_ID.

Run
Place the input CSV in the working directory, open MATLAB, run the script.

Requirements
MATLAB R2020b+.

Notes
PCA alignment is computed per cell using both channels. Photon counts are summed per longitudinal bin to form intensity profiles. Profiles are smoothed with a Gaussian kernel (sigma = 1.5 bins). The binned axis is divided into three equal-length regions and PCCs are computed for each region and the full cell.
