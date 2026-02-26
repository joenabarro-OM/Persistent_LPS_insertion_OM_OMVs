% new_lps_or_old_LPS_cluster_summary_graphs_code
%
% Purpose
% Generate replicate-aggregated summary plots (mean ± s.e.m.) for cluster
% number, mean cluster area and mean cluster density over time, split by
% region (Mid vs Pole), from the per_cell_region output of the DBSCAN pipeline.
%
% Inputs
% per_cell_region.csv produced by MATLAB_DBSCAN_Background_or_New_LPS_dSTORM_individual_cell_code
% Required columns:
% Time_min, Region, NClusters, MeanArea_nm2, MeanDensity_loc_per_nm2, Clusters_per_um2
% Optional: Replicate, Condition
%
% Outputs
% out/summary_<metric>.png and out/summary_<metric>.csv
%

clear; clc;

%% Parameters (edit)
inputFile = fullfile(pwd, 'out_dbscan_singlecolour', 'per_cell_region.csv');
outDir    = fullfile(pwd, 'out_dbscan_singlecolour', 'summary_plots');

if ~exist(outDir, 'dir'); mkdir(outDir); end

%% Load
T = readtable(inputFile);

need = {'Time_min','Region','NClusters','MeanArea_nm2','MeanDensity_loc_per_nm2','Clusters_per_um2'};
for k = 1:numel(need)
    if ~ismember(need{k}, T.Properties.VariableNames)
        error('Missing required column: %s', need{k});
    end
end

T.Region = string(T.Region);
T.Time_min = double(T.Time_min);

hasRep = ismember('Replicate', T.Properties.VariableNames);

%% Helper: compute mean ± SEM by time/region (optionally by replicate first)
function S = mean_sem_by_time_region(Tin, valueVar, hasRep)
    Tin = Tin(:, [{'Time_min','Region'}, valueVar]);
    Tin = Tin(~isnan(Tin.(valueVar)), :);

    if hasRep
        Tin.Replicate = string(Tin.Replicate);
        Tin = Tin(:, [{'Time_min','Region','Replicate'}, valueVar]);

        % replicate means first
        [G1, g1] = findgroups(Tin.Time_min, Tin.Region, Tin.Replicate);
        repMean = splitapply(@mean, Tin.(valueVar), G1);

        g1 = table(g1(:,1), g1(:,2), repMean, 'VariableNames', {'Time_min','Region','repMean'});

        [G2, g2] = findgroups(g1.Time_min, g1.Region);
        mu = splitapply(@mean, g1.repMean, G2);
        se = splitapply(@(x) std(x,0,'omitnan')/sqrt(numel(x)), g1.repMean, G2);
        n  = splitapply(@numel, g1.repMean, G2);

        S = table(g2.Time_min, g2.Region, mu, se, n, 'VariableNames', {'Time_min','Region','Mean','SEM','Nrep'});
    else
        [G, g] = findgroups(Tin.Time_min, Tin.Region);
        mu = splitapply(@mean, Tin.(valueVar), G);
        se = splitapply(@(x) std(x,0,'omitnan')/sqrt(numel(x)), Tin.(valueVar), G);
        n  = splitapply(@numel, Tin.(valueVar), G);

        S = table(g.Time_min, g.Region, mu, se, n, 'VariableNames', {'Time_min','Region','Mean','SEM','Ncells'});
    end
end

%% Metrics to plot
metrics = {
    'Clusters_per_um2', 'Clusters per \mum^2'
    'MeanArea_nm2', 'Mean cluster area (nm^2)'
    'MeanDensity_loc_per_nm2', 'Mean cluster density (loc per nm^2)'
};

for mi = 1:size(metrics,1)
    varName = metrics{mi,1};
    ylab    = metrics{mi,2};

    S = mean_sem_by_time_region(T, varName, hasRep);

    writetable(S, fullfile(outDir, "summary_" + varName + ".csv"));

    fig = figure('Color','w'); hold on;
    regions = unique(S.Region);
    for ri = 1:numel(regions)
        r = regions(ri);
        Sr = S(S.Region == r, :);
        [tSorted, order] = sort(Sr.Time_min);
        y = Sr.Mean(order);
        e = Sr.SEM(order);

        errorbar(tSorted, y, e, '-o', 'LineWidth', 1.2, 'MarkerSize', 5);
    end
    xlabel('Time (min)');
    ylabel(ylab);
    legend(regions, 'Location','best');
    box on;

    exportgraphics(fig, fullfile(outDir, "summary_" + varName + ".png"), 'Resolution', 300);
    close(fig);
end

disp("Wrote summary plots and tables to: " + outDir);
