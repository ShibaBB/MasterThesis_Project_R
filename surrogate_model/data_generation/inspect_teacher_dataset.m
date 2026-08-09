%% Surrogate Dataset Inspection
% This script performs pre-training sanity checks and visual diagnostics for
% a generated surrogate dataset.

clearvars -except surrogate_inspection_config;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
project_root = fileparts(surrogate_root);

%% Configuration
addpath(surrogate_root);
run_config = resolve_dataset_run_config();
dataset_file = run_config.paths.teacher_dataset_file;
inspection_name = sprintf('%s_dataset_inspection_%s', ...
    lower(char(string(run_config.generation.material))), char(string(run_config.run_id)));

artifacts_dir = fullfile(surrogate_root, 'data_generation', 'artifacts', inspection_name);
figures_dir = fullfile(artifacts_dir, 'figures');
artifacts_dir_overridden = false;
figures_dir_overridden = false;

num_random_curve_plots = 6;
max_histogram_features = 7;

if exist('surrogate_inspection_config', 'var')
    if isfield(surrogate_inspection_config, 'dataset_file'), dataset_file = surrogate_inspection_config.dataset_file; end
    if isfield(surrogate_inspection_config, 'inspection_name'), inspection_name = surrogate_inspection_config.inspection_name; end
    if isfield(surrogate_inspection_config, 'artifacts_dir')
        artifacts_dir = surrogate_inspection_config.artifacts_dir;
        artifacts_dir_overridden = true;
    end
    if isfield(surrogate_inspection_config, 'figures_dir')
        figures_dir = surrogate_inspection_config.figures_dir;
        figures_dir_overridden = true;
    end
    if isfield(surrogate_inspection_config, 'num_random_curve_plots'), num_random_curve_plots = surrogate_inspection_config.num_random_curve_plots; end
    if isfield(surrogate_inspection_config, 'max_histogram_features'), max_histogram_features = surrogate_inspection_config.max_histogram_features; end
end

if ~artifacts_dir_overridden
    artifacts_dir = fullfile(surrogate_root, 'data_generation', 'artifacts', inspection_name);
end

if ~figures_dir_overridden
    figures_dir = fullfile(artifacts_dir, 'figures');
end

%% Validate environment
if ~exist(dataset_file, 'file')
    error('Dataset file not found: %s', dataset_file);
end

if ~exist(artifacts_dir, 'dir')
    mkdir(artifacts_dir);
end

if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

%% Load dataset
dataset = load(dataset_file);
required_variables = {'X', 'Y', 'freq_grid', 'dataset_info', 'sample_metadata'};
for i = 1:numel(required_variables)
    if ~isfield(dataset, required_variables{i})
        error('Dataset file is missing required variable: %s', required_variables{i});
    end
end

X = dataset.X;
Y = dataset.Y;
freq_grid = dataset.freq_grid;
dataset_info = dataset.dataset_info;
sample_metadata = dataset.sample_metadata;

if size(X, 1) ~= size(Y, 1)
    error('X and Y must have the same number of rows.');
end

if size(Y, 2) ~= numel(freq_grid)
    error('The number of Y columns must match the frequency grid length.');
end

num_samples = size(X, 1);
num_features = size(X, 2);
num_outputs = size(Y, 2);

%% Compute summary statistics
feature_names = dataset_info.parameter_names;
feature_summary = table('Size', [num_features, 5], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Feature', 'Min', 'Max', 'Mean', 'Std'});

for feature_idx = 1:num_features
    feature_summary.Feature(feature_idx) = string(feature_names{feature_idx});
    feature_summary.Min(feature_idx) = min(X(:, feature_idx));
    feature_summary.Max(feature_idx) = max(X(:, feature_idx));
    feature_summary.Mean(feature_idx) = mean(X(:, feature_idx));
    feature_summary.Std(feature_idx) = std(X(:, feature_idx), 0, 1);
end

curve_summary = struct();
curve_summary.alpha_min = min(Y, [], 'all');
curve_summary.alpha_max = max(Y, [], 'all');
curve_summary.alpha_mean = mean(Y, 'all');
curve_summary.alpha_std = std(Y, 0, 'all');
curve_summary.curvewise_alpha_min = min(Y, [], 2);
curve_summary.curvewise_alpha_max = max(Y, [], 2);

porosity_labels = string({sample_metadata.porosityfolder});
unique_porosity_labels = unique(porosity_labels);
porosity_counts = zeros(numel(unique_porosity_labels), 1);
for idx = 1:numel(unique_porosity_labels)
    porosity_counts(idx) = sum(porosity_labels == unique_porosity_labels(idx));
end

inspection_summary = struct();
inspection_summary.num_samples = num_samples;
inspection_summary.num_features = num_features;
inspection_summary.num_outputs = num_outputs;
inspection_summary.feature_summary = feature_summary;
inspection_summary.curve_summary = curve_summary;
inspection_summary.unique_porosity_labels = unique_porosity_labels;
inspection_summary.porosity_counts = porosity_counts;

save(fullfile(artifacts_dir, 'dataset_inspection_summary.mat'), 'inspection_summary', '-v7.3');
write_inspection_report(fullfile(artifacts_dir, 'dataset_inspection_report.txt'), inspection_summary, dataset_info);

%% Generate plots
plot_feature_histograms(X, feature_names, figures_dir, max_histogram_features);
plot_alpha_band(freq_grid, Y, figures_dir);
plot_random_curves(freq_grid, Y, figures_dir, num_random_curve_plots);
plot_feature_correlation(X, feature_names, figures_dir);
plot_porosity_counts(unique_porosity_labels, porosity_counts, figures_dir);
plot_curve_extrema_histograms(curve_summary, figures_dir);

fprintf('Dataset inspection complete.\n');
fprintf('Samples: %d | Features: %d | Outputs: %d\n', num_samples, num_features, num_outputs);
fprintf('Inspection artifacts saved to %s\n', artifacts_dir);

%% Local functions
function write_inspection_report(report_file, inspection_summary, dataset_info)
    fid = fopen(report_file, 'w');
    if fid == -1
        error('Could not open inspection report for writing: %s', report_file);
    end

    cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, 'Surrogate Dataset Inspection Report\n');
    fprintf(fid, 'Fiber: %s\n', dataset_info.fiberfolder);
    fprintf(fid, 'Samples: %d\n', inspection_summary.num_samples);
    fprintf(fid, 'Features: %d\n', inspection_summary.num_features);
    fprintf(fid, 'Outputs: %d\n', inspection_summary.num_outputs);
    fprintf(fid, 'Frequency Range: %.2f Hz to %.2f Hz\n', dataset_info.freq_min, dataset_info.freq_max);
    fprintf(fid, 'Frequency Points: %d\n\n', dataset_info.n_freq);

    fprintf(fid, 'Feature Summary\n');
    for i = 1:height(inspection_summary.feature_summary)
        fprintf(fid, '%s | min=%.6e | max=%.6e | mean=%.6e | std=%.6e\n', ...
            inspection_summary.feature_summary.Feature(i), ...
            inspection_summary.feature_summary.Min(i), ...
            inspection_summary.feature_summary.Max(i), ...
            inspection_summary.feature_summary.Mean(i), ...
            inspection_summary.feature_summary.Std(i));
    end

    fprintf(fid, '\nAlpha Summary\n');
    fprintf(fid, 'Global alpha min  : %.6e\n', inspection_summary.curve_summary.alpha_min);
    fprintf(fid, 'Global alpha max  : %.6e\n', inspection_summary.curve_summary.alpha_max);
    fprintf(fid, 'Global alpha mean : %.6e\n', inspection_summary.curve_summary.alpha_mean);
    fprintf(fid, 'Global alpha std  : %.6e\n', inspection_summary.curve_summary.alpha_std);

    fprintf(fid, '\nPorosity Counts\n');
    for i = 1:numel(inspection_summary.unique_porosity_labels)
        fprintf(fid, '%s : %d\n', inspection_summary.unique_porosity_labels(i), inspection_summary.porosity_counts(i));
    end
end

function plot_feature_histograms(X, feature_names, figures_dir, max_histogram_features)
    num_features = min(size(X, 2), max_histogram_features);
    fig = figure('Visible', 'off', 'Color', 'w');
    tiledlayout(num_features, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:num_features
        nexttile;
        histogram(X(:, i), 30);
        xlabel(feature_names{i}, 'Interpreter', 'none');
        ylabel('Count');
        title(sprintf('Histogram: %s', feature_names{i}), 'Interpreter', 'none');
        grid on;
    end

    save_figure(fig, fullfile(figures_dir, 'feature_histograms.png'));
    close(fig);
end

function plot_alpha_band(freq_grid, Y, figures_dir)
    alpha_mean = mean(Y, 1);
    alpha_p05 = prctile(Y, 5, 1);
    alpha_p95 = prctile(Y, 95, 1);

    fig = figure('Visible', 'off', 'Color', 'w');
    fill([freq_grid, fliplr(freq_grid)], [alpha_p05, fliplr(alpha_p95)], ...
        [0.85, 0.90, 1.00], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    hold on;
    plot(freq_grid, alpha_mean, 'b-', 'LineWidth', 1.8);
    xlabel('Frequency (Hz)');
    ylabel('\alpha');
    title('Absorption Mean and 5-95% Band');
    grid on;

    save_figure(fig, fullfile(figures_dir, 'alpha_band.png'));
    close(fig);
end

function plot_random_curves(freq_grid, Y, figures_dir, num_random_curve_plots)
    n_curves = size(Y, 1);
    num_to_plot = min(num_random_curve_plots, n_curves);

    rng(2024);
    selected_idx = randperm(n_curves, num_to_plot);

    fig = figure('Visible', 'off', 'Color', 'w');
    plot(freq_grid, Y(selected_idx, :)', 'LineWidth', 1.2);
    xlabel('Frequency (Hz)');
    ylabel('\alpha');
    title('Randomly Selected Absorption Curves');
    grid on;

    save_figure(fig, fullfile(figures_dir, 'random_absorption_curves.png'));
    close(fig);
end

function plot_feature_correlation(X, feature_names, figures_dir)
    correlation_matrix = corrcoef(X);

    fig = figure('Visible', 'off', 'Color', 'w');
    imagesc(correlation_matrix);
    axis image;
    colorbar;
    clim([-1, 1]);
    xticks(1:numel(feature_names));
    yticks(1:numel(feature_names));
    xticklabels(feature_names);
    yticklabels(feature_names);
    xtickangle(30);
    title('Feature Correlation Matrix', 'Interpreter', 'none');

    save_figure(fig, fullfile(figures_dir, 'feature_correlation_matrix.png'));
    close(fig);
end

function plot_porosity_counts(unique_porosity_labels, porosity_counts, figures_dir)
    fig = figure('Visible', 'off', 'Color', 'w');
    bar(categorical(cellstr(unique_porosity_labels)), porosity_counts);
    xlabel('Porosity Folder');
    ylabel('Sample Count');
    title('Samples per Porosity Group');
    grid on;

    save_figure(fig, fullfile(figures_dir, 'porosity_counts.png'));
    close(fig);
end

function plot_curve_extrema_histograms(curve_summary, figures_dir)
    fig = figure('Visible', 'off', 'Color', 'w');
    tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    histogram(curve_summary.curvewise_alpha_min, 30);
    xlabel('Per-Curve Minimum \alpha');
    ylabel('Count');
    title('Distribution of Per-Curve Minimum Absorption');
    grid on;

    nexttile;
    histogram(curve_summary.curvewise_alpha_max, 30);
    xlabel('Per-Curve Maximum \alpha');
    ylabel('Count');
    title('Distribution of Per-Curve Maximum Absorption');
    grid on;

    save_figure(fig, fullfile(figures_dir, 'curve_extrema_histograms.png'));
    close(fig);
end

function save_figure(fig, output_file)
    if exist('exportgraphics', 'file') == 2
        exportgraphics(fig, output_file, 'Resolution', 200);
    else
        saveas(fig, output_file);
    end
end
