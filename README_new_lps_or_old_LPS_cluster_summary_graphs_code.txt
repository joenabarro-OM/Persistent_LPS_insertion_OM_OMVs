Purpose
Generate replicate-aggregated summary plots (mean ± s.e.m.) for cluster number, mean cluster area and mean cluster density over time, split by region (Mid vs Pole).

Input
per_cell_region.csv produced by MATLAB_DBSCAN_Background_or_New_LPS_dSTORM_individual_cell_code

Run
1) Ensure out_dbscan_singlecolour/per_cell_region.csv exists.
2) Run in MATLAB:
new_lps_or_old_LPS_cluster_summary_graphs_code

Outputs (written to out_dbscan_singlecolour/summary_plots/)
summary_Clusters_per_um2.csv + PNG
summary_MeanArea_nm2.csv + PNG
summary_MeanDensity_loc_per_nm2.csv + PNG

Notes
If a Replicate column is present, replicate means are computed first and then aggregated across replicates.
