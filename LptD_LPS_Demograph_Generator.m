% Script: LptD_LPS_Demograph_Generator
% Purpose: Generate demographs for LptD and newly inserted LPS along the
%  PCA-aligned long axis from two-colour dSTORM coordinates.
%
% Input (CSV):
%   LptD-newly-inserted_LPS_demograph_coordinates.csv
%   Required columns:
% - Cell_ID (or CellID)
% - Channel (LptD or LPS)
% - PositionX_nm_, PositionY_nm_
%
% Output:
% - Figure: demograph_LptD_LPS.png (optional)
% - CSV: demograph_LptD.csv, demograph_LPS.csv (optional)
%
% Notes (Methods-matching):
% - Per-cell PCA rotation uses both channels so the same axis is used for LptD and LPS.
% - Binning along x uses binWidth = 25 nm.
% - Per-cell profiles are normalised to their own maximum.
% - Rows are centre-aligned using the midpoint (median) of each cell’s localisation distribution.
% - NaNs (padding) are rendered black.
%
clear; clc;

%% Parameters
inputFile = 'LptD-newly-inserted_LPS_demograph_coordinates.csv';
binWidth_nm = 25;
saveOutputs = true;
outPrefix = 'demograph';

channels = {'LptD','LPS'};

%% Load data
T = readtable(inputFile);

cellVar = pickVar(T, {'Cell_ID','CellID'});
requireVars(T, {'Channel','PositionX_nm_','PositionY_nm_'});
cellIDs = unique(T.(cellVar));

% Precompute cell length for sorting (PCA on both channels)
cellLength = nan(numel(cellIDs),1);
rotCache = cell(numel(cellIDs),1);
centerCache = nan(numel(cellIDs),2);

for i = 1:numel(cellIDs)
    id = cellIDs(i);
    sub = T(T.(cellVar)==id, :);
    XY = [sub.PositionX_nm_, sub.PositionY_nm_];
    mu = mean(XY,1);
    XY0 = XY - mu;
    [~,~,V] = svd(XY0, 'econ');
    rotCache{i} = V;
    centerCache(i,:) = mu;
    XYr = XY0 * V;
    cellLength(i) = range(XYr(:,1));
end

[~, idx] = sort(cellLength, 'ascend');
sortedIDs = cellIDs(idx);

% First pass: build per-cell binned rows and track global width
rows = struct();
maxBins = 0;
centerBinIndex = nan(numel(sortedIDs),1);

for c = 1:numel(channels)
    rows.(channels{c}) = cell(numel(sortedIDs),1);
end

for i = 1:numel(sortedIDs)
    id = sortedIDs(i);
    sub = T(T.(cellVar)==id, :);
    V = rotCache{idx(i)};
    mu = centerCache(idx(i),:);

    XY = [sub.PositionX_nm_, sub.PositionY_nm_];
    XYr = (XY - mu) * V;
    xAll = XYr(:,1);

    % shared edges per cell (both channels)
    edges = (floor(min(xAll)/binWidth_nm)*binWidth_nm):binWidth_nm:(ceil(max(xAll)/binWidth_nm)*binWidth_nm);
    if numel(edges) < 2
        edges = [min(xAll)-binWidth_nm, max(xAll)+binWidth_nm];
    end

    % midpoint defined by median of the localisation distribution
    xMid = median(xAll);
    centerBinIndex(i) = min(max(floor((xMid - edges(1))/binWidth_nm) + 1, 1), numel(edges)-1);

    for c = 1:numel(channels)
        chan = channels{c};
        subC = sub(strcmp(sub.Channel, chan), :);
        if isempty(subC)
            rows.(chan){i} = [];
            continue;
        end
        XYc = ([subC.PositionX_nm_, subC.PositionY_nm_] - mu) * V;
        xC = XYc(:,1);

        counts = histcounts(xC, edges);
        if all(counts==0)
            rows.(chan){i} = zeros(1, numel(counts));
        else
            rows.(chan){i} = counts / max(counts); % per-cell normalisation
        end
        maxBins = max(maxBins, numel(counts));
    end
end

%% Assemble demograph matrices with centre alignment and NaN padding
midIdx = ceil(maxBins/2);
binAxis = ((1:maxBins) - midIdx) * binWidth_nm;

dem = struct();
for c = 1:numel(channels)
    dem.(channels{c}) = nan(numel(sortedIDs), maxBins);
end

for i = 1:numel(sortedIDs)
    thisCenter = centerBinIndex(i);
    for c = 1:numel(channels)
        chan = channels{c};
        row = rows.(chan){i};
        if isempty(row); continue; end
        startIdx = midIdx - thisCenter + 1;
        endIdx = startIdx + numel(row) - 1;
        if startIdx < 1
            row = row(2-startIdx:end);
            startIdx = 1;
        end
        if endIdx > maxBins
            row = row(1:(maxBins-startIdx+1));
            endIdx = maxBins;
        end
        dem.(chan)(i, startIdx:endIdx) = row;
    end
end

%% Plot
fig = figure('Color','w','Position',[100 100 1200 600]);
for c = 1:numel(channels)
    subplot(1,2,c);
    M = dem.(channels{c});
    h = imagesc(binAxis, 1:size(M,1), M);
    colormap(gca, 'parula'); caxis([0 1]);
    set(h, 'AlphaData', ~isnan(M));
    set(gca, 'Color', 'k', 'YDir','normal');
    xlabel('Long axis position (nm)');
    ylabel('Cells (sorted by length)');
    title(sprintf('%s demograph', channels{c}), 'Interpreter','none');
    colorbar;
end

if saveOutputs
    exportgraphics(fig, sprintf('%s_LptD_LPS.png', outPrefix), 'Resolution', 300);
    writematrix([binAxis; dem.LptD]', sprintf('%s_LptD.csv', outPrefix));
    writematrix([binAxis; dem.LPS]',  sprintf('%s_LPS.csv', outPrefix));
end

%% Local functions
function v = pickVar(T, candidates)
    v = '';
    for i = 1:numel(candidates)
        if ismember(candidates{i}, T.Properties.VariableNames)
            v = candidates{i};
            return;
        end
    end
    error('Missing required column: expected one of [%s]', strjoin(candidates, ', '));
end

function requireVars(T, vars)
    missing = vars(~ismember(vars, T.Properties.VariableNames));
    if ~isempty(missing)
        error('Missing required column(s): %s', strjoin(missing, ', '));
    end
end
