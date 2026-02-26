README: Python_code_New_Old_LPS_DBSCAN

Purpose
  Quantify spatial overlap between background (old) and newly inserted (new) LPS clusters in dual-colour 2D dSTORM.

Inputs
  Two CSV files:
    - background/old LPS localisations
    - newly inserted/new LPS localisations
  Required columns in each file:
    - Position X [nm]
    - Position Y [nm]

Run
  python Python_code_New_Old_LPS_DBSCAN.py --old <old.csv> --new <new.csv>

Defaults (Methods)
  DBSCAN: eps = 30 nm; min_samples = 7

Outputs
  Output folder: <old_basename>__<new_basename>/
    - LPS_output_summary.csv
    - combined_overlap_map.png
    - old_lps_cluster_histogram.csv
    - new_lps_cluster_histogram.csv

Dependencies
  Python 3.10+
  numpy, pandas, matplotlib, shapely, scipy, scikit-learn
