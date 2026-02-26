README: Python_code_phase_seperation_model_2

Purpose
  Simulation Model 2: phase separation / demixing control.

Run
  python Python_code_phase_seperation_model_2.py
  python Python_code_phase_seperation_model_2.py --seed 4

Notes
  Uses the same field geometry, global kinetics, diffusion engine, insertion schedule and OMV module as Model 1.
  Replaces insertion-trapping with void coalescence (area-preserving annealing) to generate domain coarsening.

Outputs
  omv_lps_out_model2/run_YYYYMMDD-HHMMSS/
    frames/frame_###min.png
    montage_0-120min.png
    csv/old_LPS_<t>_min.csv
    csv/new_LPS_<t>_min.csv
    csv/summary.csv

Dependencies
  Python 3.9+
  numpy, matplotlib
  tqdm (optional)
