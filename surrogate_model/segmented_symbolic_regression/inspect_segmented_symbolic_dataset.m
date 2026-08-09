%% Segmented Symbolic Dataset Inspection
% This script performs summary checks and visual diagnostics for a scalar
% segmented symbolic regression dataset.

clearvars -except symbolic_inspection_config;
clc;

script_dir = fileparts(mfilename('fullpath'));

%% Configuration
run_config = segmented_dataset_run_config();
dataset_run = run_config.dataset_run;
dataset_file = run_config.segmented_dataset_file;
inspection_name = sprintf( ...
    'wool_segmented_symbolic_dataset_inspection_%s', char(string(dataset_run)));

artifacts_dir = fullfile(script_dir, 'artifacts', inspection_name);
figures_dir = fullfile(artifacts_dir, 'figures');
artifacts_dir_overridden = false;
figures_dir_overridden = false;

max_histogram_features = 8;
scatter_max_points = 5000;

if exist('symbolic_inspection_config', 'var')
    if isfield(symbolic_inspection_config, 'dataset_run')
        run_config = segmented_dataset_run_config(symbolic_inspection_config.dataset_run);
        dataset_run = run_config.dataset_run;
        dataset_file = run_config.segmented_dataset_file;
    end
    if isfield(symbolic_inspection_config, 'dataset_file'), dataset_file = symbolic_inspection_config.dataset_file; end
    if isfield(symbolic_inspection_config, 'inspection_name'), inspection_name = symbolic_inspection_config.inspection_name; end
    if isfield(symbolic_inspection_config, 'artifacts_dir')
        artifacts_dir = symbolic_inspection_config.artifacts_dir;
        artifacts_dir_overridden = true;
    end
    if isfield(symbolic_inspection_config, 'figures_dir')
        figures_dir = symbolic_inspection_config.figures_dir;
        figures_dir_overridden = true;
    end
    if isfield(symbolic_inspection_config, 'max_histogram_features'), max_histogram_features = symbolic_inspection_config.max_histogram_features; end
    if isfield(symbolic_inspection_config, 'scatter_max_points'), scatter_max_points = symbolic_inspection_config.scatter_max_points; end
end

if ~artifacts_dir_overridden
    artifacts_dir = fullfile(script_dir, 'artifacts', inspection_name);
end

if ~figures_dir_overridden
    figures_dir = fullfile(artifacts_dir, 'figures');
end

%% Validate environment
assert_segmented_dataset_run_paths(dataset_run, dataset_file);

if ~exist(dataset_file, 'file')
    error('Symbolic dataset file not found: %s', dataset_file);
end

if ~exist(artifacts_dir, 'dir')
    mkdir(artifacts_dir);
end

if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

%% Load dataset
dataset = load(dataset_file);
required_variables = {'X_symbolic', 'y_symbolic', 'segment_index', 'segment_definitions', 'symbolic_dataset_info'};
for i = 1:numel(required_variables)
    if ~isfield(dataset, required_variables{i})
        error('Symbolic dataset is missing required variable: %s', required_variables{i});
    end
end

X_symbolic = dataset.X_symbolic;
y_symbolic = dataset.y_symbolic;
segment_index = dataset.segment_index;
segment_definitions = dataset.segment_definitions;
symbolic_dataset_info = dataset.symbolic_dataset_info;

if size(X_symbolic, 1) ~= numel(y_symbolic)
    error('X_symbolic row count must match numel(y_symbolic).');
end

if numel(segment_index) ~= numel(y_symbolic)
    error('segment_index length must match numel(y_symbolic).');
end

feature_names = symbolic_dataset_info.feature_names;
num_features = size(X_symbolic, 2);
num_samples = size(X_symbolic, 1);

%% Compute summary statistics
feature_summary = table('Size', [num_features, 5], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Feature', 'Min', 'Max', 'Mean', 'Std'});

for feature_idx = 1:num_features
    feature_summary.Feature(feature_idx) = string(feature_names{feature_idx});
    feature_summary.Min(feature_idx) = min(X_symbolic(:, feature_idx));
    feature_summary.Max(feature_idx) = max(X_symbolic(:, feature_idx));
    feature_summary.Mean(feature_idx) = mean(X_symbolic(:, feature_idx));
    feature_summary.Std(feature_idx) = std(X_symbolic(:, feature_idx), 0, 1);
end

target_summary = struct();
target_summary.alpha_min = min(y_symbolic);
target_summary.alpha_max = max(y_symbolic);
target_summary.alpha_mean = mean(y_symbolic);
target_summary.alpha_std = std(y_symbolic, 0, 1);

num_segments = numel(segment_definitions);
segment_summary = table('Size', [num_segments, 6], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Segment', 'Count', 'FrequencyMinHz', 'FrequencyMaxHz', 'AlphaMean', 'AlphaStd'});

for seg_idx = 1:num_segments
    row_mask = segment_index == seg_idx;
    segment_summary.Segment(seg_idx) = string(segment_definitions(seg_idx).name);
    segment_summary.Count(seg_idx) = sum(row_mask);
    segment_summary.FrequencyMinHz(seg_idx) = min(X_symbolic(row_mask, end));
    segment_summary.FrequencyMaxHz(seg_idx) = max(X_symbolic(row_mask, end));
    segment_summary.AlphaMean(seg_idx) = mean(y_symbolic(row_mask));
    segment_summary.AlphaStd(seg_idx) = std(y_symbolic(row_mask), 0, 1);
end

inspection_summary = struct();
inspection_summary.num_samples = num_samples;
inspection_summary.num_features = num_features;
inspection_summary.feature_summary = feature_summary;
inspection_summary.target_summary = target_summary;
inspection_summary.segment_summary = segment_summary;

save(fullfile(artifacts_dir, 'symbolic_dataset_inspection_summary.mat'), 'inspection_summary', '-v7.3');
write_inspection_report(fullfile(artifacts_dir, 'symbolic_dataset_inspection_report.txt'), inspection_summary, symbolic_dataset_info);

%% Generate plots
plot_feature_histograms(X_symbolic, feature_names, figures_dir, max_histogram_features);
plot_target_histogram(y_symbolic, figures_dir);
plot_segment_counts(segment_summary, figures_dir);
plot_alpha_vs_frequency(X_symbolic(:, end), y_symbolic, segment_index, segment_definitions, figures_dir, scatter_max_points);

fprintf('Symbolic dataset inspection complete.\n');
fprintf('Samples: %d | Features: %d\n', num_samples, num_features);
fprintf('Inspection artifacts saved to %s\n', artifacts_dir);

%% Local functions
function write_inspection_report(report_file, inspection_summary, symbolic_dataset_info)
    fid = fopen(report_file, 'w');
    if fid == -1
        error('Could not open symbolic inspection report for writing: %s', report_file);
    end

    cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, 'Symbolic Dataset Inspection Report\n');
    fprintf(fid, 'Fiber: %s\n', symbolic_dataset_info.fiberfolder);
    fprintf(fid, 'Curve Samples: %d\n', symbolic_dataset_info.num_curve_samples);
    fprintf(fid, 'Symbolic Samples: %d\n', inspection_summary.num_samples);
    fprintf(fid, 'Features: %d\n', inspection_summary.num_features);
    fprintf(fid, 'Frequency Range: %.2f Hz to %.2f Hz\n\n', ...
        symbolic_dataset_info.freq_min, symbolic_dataset_info.freq_max);

    fprintf(fid, 'Feature Summary\n');
    for i = 1:height(inspection_summary.feature_summary)
        fprintf(fid, '%s | min=%.6e | max=%.6e | mean=%.6e | std=%.6e\n', ...
            inspection_summary.feature_summary.Feature(i), ...
            inspection_summary.feature_summary.Min(i), ...
            inspection_summary.feature_summary.Max(i), ...
            inspection_summary.feature_summary.Mean(i), ...
            inspection_summary.feature_summary.Std(i));
    end

    fprintf(fid, '\nTarget Summary\n');
    fprintf(fid, 'Alpha min  : %.6e\n', inspection_summary.target_summary.alpha_min);
    fprintf(fid, 'Alpha max  : %.6e\n', inspection_summary.target_summary.alpha_max);
    fprintf(fid, 'Alpha mean : %.6e\n', inspection_summary.target_summary.alpha_mean);
    fprintf(fid, 'Alpha std  : %.6e\n', inspection_summary.target_summary.alpha_std);

    fprintf(fid, '\nSegment Summary\n');
    for i = 1:height(inspection_summary.segment_summary)
        fprintf(fid, '%s | count=%d | f_min=%.2f | f_max=%.2f | alpha_mean=%.6e | alpha_std=%.6e\n', ...
            inspection_summary.segment_summary.Segment(i), ...
            inspection_summary.segment_summary.Count(i), ...
            inspection_summary.segment_summary.FrequencyMinHz(i), ...
            inspection_summary.segment_summary.FrequencyMaxHz(i), ...
            inspection_summary.segment_summary.AlphaMean(i), ...
            inspection_summary.segment_summary.AlphaStd(i));
    end
end

function plot_feature_histograms(X_symbolic, feature_names, figures_dir, max_histogram_features)
    num_features_to_plot = min(size(X_symbolic, 2), max_histogram_features);
    fig = figure('Visible', 'off', 'Color', 'w');
    tiledlayout(num_features_to_plot, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:num_features_to_plot
        nexttile;
        histogram(X_symbolic(:, i), 30);
        xlabel(feature_names{i}, 'Interpreter', 'none');
        ylabel('Count');
        title(sprintf('Histogram: %s', feature_names{i}), 'Interpreter', 'none');
        grid on;
    end

    save_figure(fig, fullfile(figures_dir, 'feature_histograms.png'));
    close(fig);
end

function plot_target_histogram(y_symbolic, figures_dir)
    fig = figure('Visible', 'off', 'Color', 'w');
    histogram(y_symbolic, 40);
    xlabel('\alpha');
    ylabel('Count');
    title('Distribution of Symbolic Targets');
    grid on;

    save_figure(fig, fullfile(figures_dir, 'target_histogram.png'));
    close(fig);
end

function plot_segment_counts(segment_summary, figures_dir)
    fig = figure('Visible', 'off', 'Color', 'w');
    bar(categorical(cellstr(segment_summary.Segment)), segment_summary.Count);
    xlabel('Frequency Segment');
    ylabel('Sample Count');
    title('Symbolic Samples per Segment');
    grid on;

    save_figure(fig, fullfile(figures_dir, 'segment_counts.png'));
    close(fig);
end

function plot_alpha_vs_frequency(freq_values, y_symbolic, segment_index, segment_definitions, figures_dir, scatter_max_points)
    n_points = numel(y_symbolic);

    if n_points > scatter_max_points
        rng(909);
        selected_idx = randperm(n_points, scatter_max_points);
    else
        selected_idx = 1:n_points;
    end

    fig = figure('Visible', 'off', 'Color', 'w');
    hold on;

    color_map = lines(numel(segment_definitions));
    for seg_idx = 1:numel(segment_definitions)
        row_mask = selected_idx(segment_index(selected_idx) == seg_idx);
        scatter(freq_values(row_mask), y_symbolic(row_mask), 10, ...
            'MarkerFaceColor', color_map(seg_idx, :), ...
            'MarkerEdgeColor', 'none', ...
            'MarkerFaceAlpha', 0.3, ...
            'DisplayName', segment_definitions(seg_idx).name);
    end

    xlabel('Frequency (Hz)');
    ylabel('\alpha');
    title('Symbolic Samples in Frequency Space');
    legend('Location', 'best');
    grid on;

    save_figure(fig, fullfile(figures_dir, 'alpha_vs_frequency.png'));
    close(fig);
end

function save_figure(fig, output_file)
    if exist('exportgraphics', 'file') == 2
        exportgraphics(fig, output_file, 'Resolution', 200);
    else
        saveas(fig, output_file);
    end
end
