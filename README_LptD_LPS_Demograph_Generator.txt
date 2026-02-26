LptD_LPS_Demograph_Generator
Purpose
Generates demographs for LptD and newly inserted LPS along the PCA-aligned long axis across a population of cells.

Inputs
One CSV table: LptD-newly-inserted_LPS_demograph_coordinates.csv with Cell_ID (or CellID), Channel (LptD or LPS), PositionX_nm_, PositionY_nm_.

Parameters
binWidth_nm is fixed at 25 nm to match the Methods. Set saveOutputs to true/false.

Outputs
Displays the demograph figure. If saveOutputs is true, saves demograph_LptD_LPS.png and demograph_LptD.csv and demograph_LPS.csv.

Run
Place the input CSV in the working directory, open MATLAB, run the script.

Requirements
MATLAB R2020b+.

Notes
Per-cell PCA rotation is computed using both channels so LptD and LPS are displayed on the same longitudinal axis. Per-cell profiles are normalised to their own maximum. Rows are centre-aligned using the median of each cell’s localisation distribution. NaN padding is rendered black.
