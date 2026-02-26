% calculate_photon_weighted_PCCs.txt
%
% Photon-weighted Pearson correlation coefficient (PCC) between LptD* and newly inserted LPS
% from two-colour dSTORM localisation data.
%
% This script matches the Methods description:
%   - PCA alignment of each cell (long axis aligned to x)
%   - Longitudinal binning at 25 nm
%   - Photon-weighted intensity profiles (sum of photons per bin)
%   - Gaussian smoothing with sigma = 1.5 bins
%   - PCC computed for three equal-length regions (Pole 1, Mid-cell, Pole 2) and whole cell
%
% INPUT (CSV):
%   A single table containing both channels, with at minimum:
%     - CellID (or Cell_ID)
%     - Channel (e.g., 'LptD' / 'LptD*' and 'LPS' / 'newly inserted LPS')
%     - PositionX_nm_ (or PositionX_nm / PositionX [nm])
%     - PositionY_nm_ (or PositionY_nm / PositionY [nm])
%     - Photons (photon counts per localisation)
%
% OUTPUTS:
%   - PhotonWeighted_PCC_Output.csv
%       Per-cell PCC values (Pole1, Mid, Pole2, WholeCell) + basic QC metadata
%   - PhotonWeighted_PCC_Boxplot.png
%       Boxplot summarising the per-cell PCC distributions
%
% DEPENDENCIES:
%   - MATLAB R2020b+ (tested with R2024b). No toolboxes required.
%
% Author: Joe Nabarro
% Contact: joe.nabarro@york.ac.uk
% Institution: University of York
%
% -
clear; clc;

% USER SETTINGS (edit as needed)
inputCsv    = 'LptD_new_LPS_PCC_input_file.csv';
outputCsv   = 'PhotonWeighted_LptD_new_LPS_PCC_Output.csv';
outputPlot  = 'PhotonWeighted_LptD_new_LPS_PCC_Boxplot.png';

binWidth_nm        = 25;     % longitudinal bin width (nm)
smoothSigma_bins   = 1.5;    % Gaussian smoothing (sigma, in bins) [Methods: sigma = 1.5 bins]
minBinsPerRegion   = 5;      % QC: minimum bins per region required to report PCC (otherwise NaN)

% Channel label matching. 
lptdTokens = {'lptd', 'lptd*', 'lptd_star'};
lpsTokens  = {'lps', 'new', 'newly inserted'};

%% LOAD AND STANDARDISE INPUT TABLE
Traw = readtable(inputCsv);

T = standardiseInputTable(Traw);

% Unique cell IDs (preserve numeric IDs if present; otherwise treat as strings)
cellIDs = unique(T.CellID, 'stable');

% Preallocate results
nCells = numel(cellIDs);
Pole1_PCC     = nan(nCells,1);
Mid_PCC       = nan(nCells,1);
Pole2_PCC     = nan(nCells,1);
WholeCell_PCC = nan(nCells,1);
nBinsTotal    = nan(nCells,1);
CellID_out    = cell(nCells,1);

%% MAIN LOOP OVER CELLS
for i = 1:nCells
    cid = cellIDs(i);
    subT = T(T.CellID == cid, :);
    CellID_out{i} = string(cid);

    % --- PCA ALIGNMENT (via SVD): long axis -> x
    coords = [subT.X_nm, subT.Y_nm];
    coords = coords - mean(coords, 1, 'omitnan');
    [~, ~, V] = svd(coords, 'econ');
    xrot = coords * V(:,1);  % first principal axis

    % --- Build common bin edges across both channels for this cell
    xmin = floor(min(xrot, [], 'omitnan') / binWidth_nm) * binWidth_nm;
    xmax = ceil(max(xrot,  [], 'omitnan') / binWidth_nm) * binWidth_nm;
    if ~isfinite(xmin) || ~isfinite(xmax) || xmin == xmax
        continue;
    end
    edges = xmin:binWidth_nm:xmax;
    if numel(edges) < (3*minBinsPerRegion + 1)
        % Not enough bins to support 3 regions with QC threshold
        nBinsTotal(i) = numel(edges) - 1;
        continue;
    end

    % Photon-weighted longitudinal profiles (sum photons per bin)
    % Identify channels for this cell
    chanStr = lower(string(subT.Channel));

    isLptD = false(height(subT),1);
    for t = 1:numel(lptdTokens)
        isLptD = isLptD | contains(chanStr, lptdTokens{t});
    end

    isLPS = false(height(subT),1);
    for t = 1:numel(lpsTokens)
        isLPS = isLPS | contains(chanStr, lpsTokens{t});
    end

    % Weighted histograms
    profLptD = histcounts(xrot(isLptD), edges, 'Weights', subT.Photons(isLptD));
    profLPS  = histcounts(xrot(isLPS),  edges, 'Weights', subT.Photons(isLPS));

    % Smooth to reduce noise (Gaussian kernel; sigma in bins)
    profLptD = gaussianSmooth1D(profLptD, smoothSigma_bins);
    profLPS  = gaussianSmooth1D(profLPS,  smoothSigma_bins);

    L = numel(profLptD);
    nBinsTotal(i) = L;

    % QC: require adequate bins (already checked), but also avoid trivial all-zero cases
    if all(profLptD == 0) || all(profLPS == 0)
        continue;
    end

    %  Define regions: three equal-length segments along the binned profile
    idx1 = floor(L/3);
    idx2 = floor(2*L/3);

    R1 = 1:idx1;
    R2 = (idx1+1):idx2;
    R3 = (idx2+1):L;

    if numel(R1) < minBinsPerRegion || numel(R2) < minBinsPerRegion || numel(R3) < minBinsPerRegion
        continue;
    end

    %  Compute Pearson correlations (photon-weighted profiles)
    Pole1_PCC(i)     = corr(profLptD(R1)', profLPS(R1)', 'Type','Pearson', 'Rows','complete');
    Mid_PCC(i)       = corr(profLptD(R2)', profLPS(R2)', 'Type','Pearson', 'Rows','complete');
    Pole2_PCC(i)     = corr(profLptD(R3)', profLPS(R3)', 'Type','Pearson', 'Rows','complete');
    WholeCell_PCC(i) = corr(profLptD(:),   profLPS(:),   'Type','Pearson', 'Rows','complete');
end

%% SAVE OUTPUT TABLE
PCC_table = table( ...
    string(CellID_out), Pole1_PCC, Mid_PCC, Pole2_PCC, WholeCell_PCC, nBinsTotal, ...
    'VariableNames', {'CellID','Pole1_PCC','Mid_PCC','Pole2_PCC','WholeCell_PCC','nBinsTotal'} );

writetable(PCC_table, outputCsv);

%% OPTIONAL: SUMMARY BOXPLOT (distribution-style plot)
makePCCBoxplot(PCC_table, outputPlot);

disp("Done: PCCs saved to " + outputCsv);
disp("Boxplot saved to " + outputPlot);



function T = standardiseInputTable(Traw)
%STANDARDISEINPUTTABLE Harmonise common column-name variants into a standard schema.

    vars = string(Traw.Properties.VariableNames);

    % Cell ID
    cellVar = pickVar(vars, ["CellID","Cell_ID","cellID","cell_id"]);
    if cellVar == ""
        error("Input table is missing a CellID/Cell_ID column.");
    end

    % Channel
    chanVar = pickVar(vars, ["Channel","channel","Chan","chan"]);
    if chanVar == ""
        error("Input table is missing a Channel column.");
    end

    % X / Y (nm)
    xVar = pickVar(vars, ["PositionX_nm_","PositionX_nm","PositionX__nm_","PositionX [nm]","X_nm","X"]);
    yVar = pickVar(vars, ["PositionY_nm_","PositionY_nm","PositionY__nm_","PositionY [nm]","Y_nm","Y"]);
    if xVar == "" || yVar == ""
        error("Input table is missing PositionX/PositionY columns in nm.");
    end

    % Photons
    phVar = pickVar(vars, ["Photons","photons","Photon","photon_count","Intensity","photons_"]);
    if phVar == ""
        error("Input table is missing a Photons column.");
    end

    % Build standardised table
    T = table;
    T.CellID   = Traw.(cellVar);
    T.Channel  = string(Traw.(chanVar));
    T.X_nm     = double(Traw.(xVar));
    T.Y_nm     = double(Traw.(yVar));
    T.Photons  = double(Traw.(phVar));

    % Normalise CellID to a categorical-like type usable in == comparisons
    if iscell(T.CellID) || isstring(T.CellID) || ischar(T.CellID)
        T.CellID = string(T.CellID);
    end
end

function v = pickVar(vars, candidates)
%PICKVAR Return the first matching variable name from candidates, else "".
    v = "";
    for k = 1:numel(candidates)
        idx = find(strcmpi(vars, candidates(k)), 1, 'first');
        if ~isempty(idx)
            v = vars(idx);
            return;
        end
    end
end

function y = gaussianSmooth1D(x, sigmaBins)
%GAUSSIANSMOOTH1D 1D Gaussian smoothing using convolution (toolbox-free).
    x = double(x(:))'; % row vector
    if ~(isfinite(sigmaBins) && sigmaBins > 0)
        y = x;
        return;
    end
    halfWidth = max(1, ceil(4*sigmaBins));
    t = -halfWidth:halfWidth;
    k = exp(-0.5*(t./sigmaBins).^2);
    k = k ./ sum(k);
    y = conv(x, k, 'same');
end

function makePCCBoxplot(PCC_table, outPng)
%MAKEPCCBOXPLOT Save a simple boxplot summary (no custom styling).
    dataMat = [PCC_table.Pole1_PCC, PCC_table.Mid_PCC, PCC_table.Pole2_PCC, PCC_table.WholeCell_PCC];

    fig = figure('Color','w','Units','centimeters','Position',[2 2 14 10]);
    boxplot(dataMat, 'Labels', {'Pole 1','Mid-cell','Pole 2','Whole cell'});
    ylabel('Photon-weighted PCC (Pearson r)');
    set(gca, 'TickDir','out', 'Box','off');
    ylim([-1 1]);
    exportgraphics(fig, outPng, 'Resolution', 600);
    close(fig);
end
