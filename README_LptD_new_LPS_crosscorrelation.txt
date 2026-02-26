LptD_new_LPS_crosscorrelation
Purpose
Computes the radial cross-correlation g(r) between LptD and newly inserted LPS localisations for a single cell, normalised to a null model.

Inputs
Two CSV tables (one per channel) containing PositionX_nm_ and PositionY_nm_. Optional column Precision_nm_. Example inputs: LPS_647.csv and LptD_488.csv (or LPS_488.csv and LptD_647.csv).

Parameters
Edit the file names (lptdFile, lpsFile) and cellTag at the top of the script. Bin size is estimated from Precision_nm_ when present; otherwise a size-based fallback is used. Null model boundary is controlled by boundaryShrink.

Outputs
Writes/updates Individual_cross_correlation_data_sets.csv in crosscorr_output and saves gr_profile_<cellTag>.png and gr_map_<cellTag>.png.

Run
Place the two CSV files in the working directory, open MATLAB, run the script. Repeat per cell and update cellTag each time to append rows cleanly.

Requirements
MATLAB R2020b+ .

Notes
Coordinates are PCA-aligned using both channels so the long axis is x. The null model randomises LPS points within the observed cell envelope boundary to remove true co-localisation while retaining geometry and point density.
