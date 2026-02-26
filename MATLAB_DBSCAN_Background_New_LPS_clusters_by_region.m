% MATLAB_DBSCAN_Background_New_LPS_clusters_by_region
%
% Purpose
% Region-resolved single-cell cluster maps from DBSCAN per-cluster outputs.
% Cluster marker sizes are globally scaled by sqrt(area) to enable direct
% visual comparison across cells and timepoints. Poles are defined as
% |x| >= 1.5 * radius in the aligned, centred frame.
%
% Input
% per_cluster.csv produced by MATLAB_DBSCAN_Background_or_New_LPS_dSTORM_individual_cell_code
% Required columns:
% CellID, Time_min, CentroidX_nm, CentroidY_nm, Area_nm2, Region
% Optional: Replicate, Condition
%
% Output
% One PNG per CellID x Time_min in outDir.
%

clear; clc;

%% Parameters (edit)
inputFile = fullfile(pwd, 'out_dbscan_singlecolour', 'per_cluster.csv');
outDir    = fullfile(pwd, 'out_dbscan_singlecolour', 'region_maps');
minMarker = 10;
maxMarker = 200;

if ~exist(outDir,'dir'); mkdir(outDir); end

%% Load
T = readtable(inputFile);

need = {'CellID','Time_min','CentroidX_nm','CentroidY_nm','Area_nm2','Region'};
for k = 1:numel(need)
    if ~ismember(need{k}, T.Properties.VariableNames)
        error('Missing required column: %s', need{k});
    end
end

T.CellID = string(T.CellID);
T.Region = string(T.Region);

% global size scaling (sqrt area)
A = double(T.Area_nm2);
sqrtA = sqrt(A);
sqrtA(~isfinite(sqrtA)) = NaN;
aMin = min(sqrtA, [], 'omitnan');
aMax = max(sqrtA, [], 'omitnan');

scaleSize = @(s) minMarker + (maxMarker-minMarker) * (s - aMin) ./ max(aMax-aMin, eps);

%% Group by cell/time 
keys = {'CellID','Time_min'};
hasRep = ismember('Replicate', T.Properties.VariableNames);
hasCond = ismember('Condition', T.Properties.VariableNames);
if hasRep,  keys{end+1} = 'Replicate'; end
if hasCond, keys{end+1} = 'Condition'; end

[G, groupTable] = findgroups(T(:, keys));

for gi = 1:height(groupTable)
    idx = (G == gi);
    sub = T(idx,:);

    cid = string(sub.CellID(1));
    tmin = double(sub.Time_min(1));

    x = double(sub.CentroidX_nm);
    y = double(sub.CentroidY_nm);
    s = scaleSize(sqrt(double(sub.Area_nm2)));
    r = string(sub.Region);

    fig = figure('Color','w'); hold on;

    % plot by region
    regions = unique(r);
    for ri = 1:numel(regions)
        rr = regions(ri);
        m = (r == rr);
        scatter(x(m), y(m), s(m), 'filled', 'MarkerFaceAlpha', 0.85, 'MarkerEdgeAlpha', 0.85);
    end

    xlabel('Aligned X (nm)');
    ylabel('Aligned Y (nm)');
    axis equal;
    box on;
    legend(regions, 'Location','best');

    title(sprintf('Cell %s, %g min', cid, tmin), 'Interpreter','none');

    % save
    safeID = regexprep(cid, '[^\w]', '_');
    f = sprintf('Cell_%s_%gmin.png', safeID, tmin);
    exportgraphics(fig, fullfile(outDir, f), 'Resolution', 300);
    close(fig);
end

disp("Wrote region maps to: " + outDir);
