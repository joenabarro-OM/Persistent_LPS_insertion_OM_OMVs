Purpose
Single-colour dSTORM DBSCAN clustering pipeline for LPS datasets.

Inputs
A single localisation table (CSV/XLSX) with per-localisation coordinates and per-cell dimensions.
Required columns:
CellID, Time_min, PositionX_nm_, PositionY_nm_, CellLength_nm, CellRadius_nm
Optional columns:
Replicate, Condition

Methods-matching parameters (defaults in script)
DBSCAN epsilon: 30 nm
DBSCAN minPts: 7
Pole definition: |x| >= 1.5 * radius
Edge exclusion (projection z): keep points with z >= 0.2 * radius

Run
1) Place your localisation table in the working directory.
2) Open MATLAB and run:
 MATLAB_DBSCAN_Background_or_New_LPS_dSTORM_individual_cell_code
3) Edit the variable mapping block if your column names differ.

Outputs (written to out_dbscan_singlecolour/)
per_cluster.csv
One row per cluster with centroid, area, localisation density and region label.
per_cell_region.csv
One row per cell x region (Mid vs Pole) x time with cluster counts and mean properties.

Notes
This script implements the pipeline described in the Methods for single-colour DBSCAN clustering and region-resolved quantification from aligned, projected half-rod geometry.
