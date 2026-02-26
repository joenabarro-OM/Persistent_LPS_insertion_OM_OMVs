Purpose
Region-resolved single-cell cluster maps from DBSCAN per-cluster outputs, using global sqrt(area) marker-size scaling to enable direct comparison across cells and timepoints.

Input
per_cluster.csv produced by MATLAB_DBSCAN_Background_or_New_LPS_dSTORM_individual_cell_code

Run
1) Ensure out_dbscan_singlecolour/per_cluster.csv exists.
2) Run in MATLAB:
MATLAB_DBSCAN_Background_New_LPS_clusters_by_region

Outputs (written to out_dbscan_singlecolour/region_maps/)
One PNG per CellID x Time_min.

Notes
Region labels (Mid vs Pole) are taken from per_cluster.csv (computed using the Methods pole definition).
