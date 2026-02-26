% MATLAB_DBSCAN_Background_or_New_LPS_dSTORM_individual_cell_code
%
% Purpose
%   Single-colour dSTORM DBSCAN clustering pipeline for LPS datasets.
%   Per-cell PCA alignment, capsule (rod + hemispheres) masking, half-rod surface projection,
%   DBSCAN clustering, convex-hull area and per-cluster localisation-density quantification,
%   plus per-cell/per-region summary outputs.
%
% Methods match
%   DBSCAN: eps = 30 nm, minPts = 7
%   Pole definition: |x| >= 1.5 * radius (x is cluster centroid in aligned, centred frame)
%   Exclude clusters near projected edge: z < 0.2 * radius (projection-derived z)
%
% Inputs (edit the variable mapping section if needed)
%   A single localisation table (CSV/XLSX) containing, at minimum:
%     CellID, Time_min, PositionX_nm_, PositionY_nm_, CellLength_nm, CellRadius_nm
%   Optional: Replicate, Condition
%
% Outputs
%   out/per_cluster.csv  : one row per cluster
%   out/per_cell_region.csv : one row per cell x region (Mid vs Pole) x time (and replicate/condition if present)
%
% Requirements
%   MATLAB R2021b+ (DBSCAN requires Statistics and Machine Learning Toolbox)
%
% Notes
%   This script is intentionally minimal and assumes inputs are already per-cell segmented
%   and centred to the cell mid-point in the raw coordinate system. PCA alignment and
%   centring are applied per cell/time group.
%

clear; clc;

%% Parameters (edit)
inputFile = 'single_colour_localisations.csv';  % CSV or XLSX
outDir    = fullfile(pwd, 'out_dbscan_singlecolour');
eps_nm    = 30;   % DBSCAN epsilon (nm)
minPts    = 7;    % DBSCAN minimum points
poleMult  = 1.5;  % poles defined as |x| >= poleMult*R
zMinFrac  = 0.2;  % exclude projected-edge points with z < zMinFrac*R
maskPad_nm = 0;   % optional padding for capsule mask (nm)

if ~exist(outDir, 'dir'); mkdir(outDir); end

%% Load table
[~,~,ext] = fileparts(inputFile);
if any(strcmpi(ext, {'.xlsx','.xls'}))
    T = readtable(inputFile);
else
    T = readtable(inputFile);
end

%% Variable mapping (edit if your headers differ)
colCellID  = 'CellID';
colTime    = 'Time_min';
colX       = 'PositionX_nm_';
colY       = 'PositionY_nm_';
colLen     = 'CellLength_nm';
colRad     = 'CellRadius_nm';
colRep     = '';  % e.g. 'Replicate'  (leave '' if not present)
colCond    = '';  % e.g. 'Condition'  (leave '' if not present)

% basic checks
req = {colCellID,colTime,colX,colY,colLen,colRad};
for k = 1:numel(req)
    if ~ismember(req{k}, T.Properties.VariableNames)
        error('Missing required column: %s', req{k});
    end
end

% normalise types
T.(colCellID) = string(T.(colCellID));
T.(colTime)   = double(T.(colTime));

hasRep  = ~isempty(colRep)  && ismember(colRep,  T.Properties.VariableNames);
hasCond = ~isempty(colCond) && ismember(colCond, T.Properties.VariableNames);

%% Grouping keys
keys = [colCellID, colTime];
if hasRep,  keys = [keys, colRep];  end
if hasCond, keys = [keys, colCond]; end

[G, groupTable] = findgroups(T(:, keys));

% Preallocate collectors
clusterRows = {};
cellRegionRows = {};

clusterRowHdr = {'CellID','Time_min','Replicate','Condition','ClusterID', ...
    'CentroidX_nm','CentroidY_nm','NLocalisations','Area_nm2','Density_loc_per_nm2','Zmean_nm','Region'};
cellRegionHdr = {'CellID','Time_min','Replicate','Condition','Region', ...
    'NClusters','MeanArea_nm2','MeanDensity_loc_per_nm2','SurfaceArea_um2','Clusters_per_um2'};

%% Main loop
nGroups = max(G);
for gi = 1:nGroups
    idx = (G == gi);
    sub = T(idx, :);

    cid = string(sub.(colCellID)(1));
    tmin = double(sub.(colTime)(1));

    repVal  = "";
    condVal = "";
    if hasRep,  repVal  = string(sub.(colRep)(1));  end
    if hasCond, condVal = string(sub.(colCond)(1)); end

    Xraw = double(sub.(colX));
    Yraw = double(sub.(colY));

    % Remove NaNs
    ok = ~(isnan(Xraw) | isnan(Yraw));
    Xraw = Xraw(ok);
    Yraw = Yraw(ok);
    if numel(Xraw) < minPts
        continue
    end

    % Per-cell dimensions
    L = double(sub.(colLen)(1));
    R = double(sub.(colRad)(1));
    if ~(isfinite(L) && isfinite(R) && L > 0 && R > 0)
        continue
    end

    %% PCA alignment (per cell/time)
    XY = [Xraw - mean(Xraw), Yraw - mean(Yraw)];
    [~, score] = pca(XY);
    X = score(:,1);  % long axis
    Y = score(:,2);  % short axis
    % centre long-axis at 0
    X = X - mean(X);
    Y = Y - mean(Y);

    %% Capsule mask (rod + hemispheres) in aligned frame
    Lcyl = max(L - 2*R, 0);
    halfCyl = Lcyl/2;
    absX = abs(X);
    absY = abs(Y);

    inCyl = (absX <= halfCyl + maskPad_nm) & (absY <= R + maskPad_nm);

    dx = max(absX - halfCyl, 0);
    inCaps = (dx.^2 + absY.^2) <= (R + maskPad_nm).^2;

    inMask = inCyl | inCaps;

    X = X(inMask);
    Y = Y(inMask);

    if numel(X) < minPts
        continue
    end

    %% Half-rod surface projection proxy: z = sqrt(R^2 - y^2)
    z = sqrt(max(R^2 - Y.^2, 0));
    keepZ = z >= (zMinFrac * R);
    X = X(keepZ);
    Y = Y(keepZ);
    z = z(keepZ);

    if numel(X) < minPts
        continue
    end

    %% DBSCAN clustering in 2D aligned frame
    labels = dbscan([X Y], eps_nm, minPts);

    % cluster IDs (exclude noise label -1)
    clustIDs = unique(labels);
    clustIDs = clustIDs(clustIDs > 0);
    if isempty(clustIDs)
        continue
    end

    %% Surface area for half-rod (nm^2 -> um^2)
    SA_half_nm2 = (pi * R * Lcyl) + (2 * pi * R^2);
    SA_half_um2 = SA_half_nm2 / 1e6;

    %% Build per-cluster outputs
    perClust = struct('Region',{},'Area',{},'Density',{},'Cx',{},'Cy',{});
    for ci = 1:numel(clustIDs)
        id = clustIDs(ci);
        pts = (labels == id);

        Xc = X(pts);
        Yc = Y(pts);
        Zc = z(pts);
        nLoc = numel(Xc);

        cx = mean(Xc);
        cy = mean(Yc);
        zmean = mean(Zc);

        % convex hull area (nm^2)
        area_nm2 = NaN;
        if nLoc >= 3
            try
                k = convhull(Xc, Yc);
                area_nm2 = polyarea(Xc(k), Yc(k));
            catch
                area_nm2 = NaN;
            end
        end

        dens = NaN;
        if isfinite(area_nm2) && area_nm2 > 0
            dens = nLoc / area_nm2;
        end

        % region label (Mid vs Poles combined later)
        if cx <= -(poleMult * R) || cx >= (poleMult * R)
            region = "Pole";
        else
            region = "Mid";
        end

        perClust(ci).Region  = region;
        perClust(ci).Area    = area_nm2;
        perClust(ci).Density = dens;
        perClust(ci).Cx      = cx;
        perClust(ci).Cy      = cy;

        clusterRows(end+1, :) = {cid, tmin, repVal, condVal, id, cx, cy, nLoc, area_nm2, dens, zmean, region}; %#ok<AGROW>
    end

    %% Per-cell x region summary (Mid vs Pole)
    regions = ["Mid","Pole"];
    for rgi = 1:numel(regions)
        rname = regions(rgi);
        maskR = arrayfun(@(s) s.Region == rname, perClust);
        if ~any(maskR)
            nCl = 0;
            meanA = NaN;
            meanD = NaN;
        else
            nCl = sum(maskR);
            meanA = mean([perClust(maskR).Area], 'omitnan');
            meanD = mean([perClust(maskR).Density], 'omitnan');
        end

        clusters_per_um2 = nCl / SA_half_um2;

        cellRegionRows(end+1, :) = {cid, tmin, repVal, condVal, rname, nCl, meanA, meanD, SA_half_um2, clusters_per_um2}; %#ok<AGROW>
    end
end

%% Write outputs
clusterOut = cell2table(clusterRows, 'VariableNames', clusterRowHdr);
cellRegOut = cell2table(cellRegionRows, 'VariableNames', cellRegionHdr);

writetable(clusterOut, fullfile(outDir, 'per_cluster.csv'));
writetable(cellRegOut, fullfile(outDir, 'per_cell_region.csv'));

disp("Wrote: " + fullfile(outDir, 'per_cluster.csv'));
disp("Wrote: " + fullfile(outDir, 'per_cell_region.csv'));
