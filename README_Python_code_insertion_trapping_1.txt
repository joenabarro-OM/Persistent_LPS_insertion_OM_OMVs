README: Python_code_insertion_trapping_1

Purpose
  Simulation Model 1: insertion–trapping n (no phase separation).

Run
  python Python_code_insertion_trapping_1.py
  python Python_code_insertion_trapping_1.py --seed 4

Inputs
  Parameters are set at the top of the file (field size, kinetics, diffusion, OMV rules).

Outputs
  omv_lps_out_model1/run_YYYYMMDD-HHMMSS/
    frames/frame_###min.png
    montage_0-120min.png
    csv/old_LPS_<t>_min.csv
    csv/new_LPS_<t>_min.csv
    csv/summary.csv

Dependencies
  Python 3.9+
  numpy, matplotlib
  tqdm (optional)
