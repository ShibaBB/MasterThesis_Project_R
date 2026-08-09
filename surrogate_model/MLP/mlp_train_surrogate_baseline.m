%% Surrogate Baseline Training
% This script trains a first multi-output neural-network surrogate for the
% JCAL MATLAB teacher model. It loads a generated dataset, applies
% reproducible preprocessing, trains a baseline MLP, evaluates on a held-out
% test set, and exports diagnostic plots.

clearvars -except surrogate_training_config;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
project_root = fileparts(surrogate_root);

%% Configuration
addpath(surrogate_root);
run_config = resolve_dataset_run_config();
dataset_file = run_config.paths.teacher_dataset_file;
shared_split_file = run_config.paths.shared_split_file;

experiment_name = sprintf('%s_baseline', char(string(run_config.run_id)));
artifacts_root = fullfile(surrogate_root, 'MLP', 'artifacts', 'wool_baseline_mlp_runs');
artifacts_dir = '';
figures_dir = fullfile(artifacts_dir, 'figures');
artifacts_dir_overridden = false;
figures_dir_overridden = false;

apply_log10_to_inputs = true;
log10_feature_indices = [3, 5, 6, 7];
standardize_inputs = true;
standardize_outputs = true;

hidden_layer_sizes = [128, 128, 64];
activation_name = 'relu';
dropout_probability = 0.0;

max_epochs = 400;
mini_batch_size = 64;
initial_learning_rate = 1e-3;
validation_patience = 25;
execution_environment = 'auto';
training_seed = 321;

num_random_curve_plots = 5;
num_worst_case_plots = 5;
scatter_max_points = 3000;

if exist('surrogate_training_config', 'var')
    if isfield(surrogate_training_config, 'dataset_file'), dataset_file = surrogate_training_config.dataset_file; end
    if isfield(surrogate_training_config, 'shared_split_file'), shared_split_file = surrogate_training_config.shared_split_file; end
    if isfield(surrogate_training_config, 'experiment_name'), experiment_name = surrogate_training_config.experiment_name; end
    if isfield(surrogate_training_config, 'artifacts_root'), artifacts_root = surrogate_training_config.artifacts_root; end
    if isfield(surrogate_training_config, 'artifacts_dir')
        artifacts_dir = surrogate_training_config.artifacts_dir;
        artifacts_dir_overridden = true;
    end
    if isfield(surrogate_training_config, 'figures_dir')
        figures_dir = surrogate_training_config.figures_dir;
        figures_dir_overridden = true;
    end
    if isfield(surrogate_training_config, 'apply_log10_to_inputs'), apply_log10_to_inputs = surrogate_training_config.apply_log10_to_inputs; end
    if isfield(surrogate_training_config, 'log10_feature_indices'), log10_feature_indices = surrogate_training_config.log10_feature_indices; end
    if isfield(surrogate_training_config, 'standardize_inputs'), standardize_inputs = surrogate_training_config.standardize_inputs; end
    if isfield(surrogate_training_config, 'standardize_outputs'), standardize_outputs = surrogate_training_config.standardize_outputs; end
    if isfield(surrogate_training_config, 'hidden_layer_sizes'), hidden_layer_sizes = surrogate_training_config.hidden_layer_sizes; end
    if isfield(surrogate_training_config, 'activation_name'), activation_name = surrogate_training_config.activation_name; end
    if isfield(surrogate_training_config, 'dropout_probability'), dropout_probability = surrogate_training_config.dropout_probability; end
    if isfield(surrogate_training_config, 'max_epochs'), max_epochs = surrogate_training_config.max_epochs; end
    if isfield(surrogate_training_config, 'mini_batch_size'), mini_batch_size = surrogate_training_config.mini_batch_size; end
    if isfield(surrogate_training_config, 'initial_learning_rate'), initial_learning_rate = surrogate_training_config.initial_learning_rate; end
    if isfield(surrogate_training_config, 'validation_patience'), validation_patience = surrogate_training_config.validation_patience; end
    if isfield(surrogate_training_config, 'execution_environment'), execution_environment = surrogate_training_config.execution_environment; end
    if isfield(surrogate_training_config, 'training_seed'), training_seed = surrogate_training_config.training_seed; end
    if isfield(surrogate_training_config, 'num_random_curve_plots'), num_random_curve_plots = surrogate_training_config.num_random_curve_plots; end
    if isfield(surrogate_training_config, 'num_worst_case_plots'), num_worst_case_plots = surrogate_training_config.num_worst_case_plots; end
    if isfield(surrogate_training_config, 'scatter_max_points'), scatter_max_points = surrogate_training_config.scatter_max_points; end
end

if ~artifacts_dir_overridden
    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    run_folder_name = sprintf('%s_%s', timestamp, sanitize_run_name(experiment_name));
    artifacts_dir = unique_run_directory(artifacts_root, run_folder_name);
elseif directory_contains_files(artifacts_dir)
    error(['Refusing to overwrite an existing MLP artifact directory: %s. ', ...
           'Choose a new artifacts_dir or omit it to use an automatic unique run directory.'], ...
          artifacts_dir);
end

if ~figures_dir_overridden
    figures_dir = fullfile(artifacts_dir, 'figures');
end

%% Validate environment
required_functions = {'trainNetwork', 'trainingOptions', 'featureInputLayer'};
for i = 1:numel(required_functions)
    if exist(required_functions{i}, 'file') ~= 2
        error(['Required MATLAB Deep Learning Toolbox function not found: %s. ', ...
               'This training script depends on Deep Learning Toolbox.'], required_functions{i});
    end
end

if ~exist(dataset_file, 'file')
    error('Dataset file not found: %s', dataset_file);
end

if ~exist(shared_split_file, 'file')
    error('Shared curve split file not found: %s', shared_split_file);
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
    error('The number of output columns in Y must match numel(freq_grid).');
end

num_samples = size(X, 1);
num_features = size(X, 2);
num_outputs = size(Y, 2);

%% Split dataset
[split_indices, shared_split_info] = load_shared_curve_split( ...
    shared_split_file, num_samples, dataset_file);

X_train_raw = X(split_indices.train, :);
Y_train_raw = Y(split_indices.train, :);

X_validation_raw = X(split_indices.validation, :);
Y_validation_raw = Y(split_indices.validation, :);

X_test_raw = X(split_indices.test, :);
Y_test_raw = Y(split_indices.test, :);

%% Preprocess inputs and outputs
preprocessing = struct();
preprocessing.apply_log10_to_inputs = apply_log10_to_inputs;
preprocessing.log10_feature_indices = log10_feature_indices;
preprocessing.standardize_inputs = standardize_inputs;
preprocessing.standardize_outputs = standardize_outputs;

[X_train_processed, input_preprocessing] = fit_transform_inputs( ...
    X_train_raw, apply_log10_to_inputs, log10_feature_indices, standardize_inputs);
X_validation_processed = transform_inputs(X_validation_raw, input_preprocessing);
X_test_processed = transform_inputs(X_test_raw, input_preprocessing);

[Y_train_processed, output_preprocessing] = fit_standardization(Y_train_raw, standardize_outputs);
Y_validation_processed = apply_standardization(Y_validation_raw, output_preprocessing);
Y_test_processed = apply_standardization(Y_test_raw, output_preprocessing);

preprocessing.input = input_preprocessing;
preprocessing.output = output_preprocessing;

%% Build network
layers = build_mlp_layers(num_features, num_outputs, hidden_layer_sizes, activation_name, dropout_probability);

%% Train model
rng(training_seed);
options = trainingOptions('adam', ...
    'InitialLearnRate', initial_learning_rate, ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_validation_processed, Y_validation_processed}, ...
    'ValidationFrequency', max(1, floor(size(X_train_processed, 1) / mini_batch_size)), ...
    'ValidationPatience', validation_patience, ...
    'ExecutionEnvironment', execution_environment, ...
    'Verbose', true, ...
    'Plots', 'none');

[net, training_info] = trainNetwork(X_train_processed, Y_train_processed, layers, options);

%% Evaluate model
Y_train_pred_processed = predict(net, X_train_processed);
Y_validation_pred_processed = predict(net, X_validation_processed);
Y_test_pred_processed = predict(net, X_test_processed);

Y_train_pred = invert_output_standardization(Y_train_pred_processed, output_preprocessing);
Y_validation_pred = invert_output_standardization(Y_validation_pred_processed, output_preprocessing);
Y_test_pred = invert_output_standardization(Y_test_pred_processed, output_preprocessing);

metrics = struct();
metrics.train = compute_regression_metrics(Y_train_raw, Y_train_pred);
metrics.validation = compute_regression_metrics(Y_validation_raw, Y_validation_pred);
metrics.test = compute_regression_metrics(Y_test_raw, Y_test_pred);

test_curve_mae = mean(abs(Y_test_pred - Y_test_raw), 2);
[sorted_test_mae, worst_case_order] = sort(test_curve_mae, 'descend');

evaluation = struct();
evaluation.metrics = metrics;
evaluation.test_curve_mae = test_curve_mae;
evaluation.worst_case_order = worst_case_order;
evaluation.sorted_test_mae = sorted_test_mae;
evaluation.split_indices = split_indices;
evaluation.shared_split_info = shared_split_info;

%% Save artifacts
model_artifact = struct();
model_artifact.net = net;
model_artifact.training_info = training_info;
model_artifact.preprocessing = preprocessing;
model_artifact.metrics = metrics;
model_artifact.dataset_file = dataset_file;
model_artifact.dataset_info = dataset_info;
model_artifact.experiment_name = experiment_name;
model_artifact.artifacts_dir = artifacts_dir;
model_artifact.split_indices = split_indices;
model_artifact.shared_split_info = shared_split_info;
model_artifact.hidden_layer_sizes = hidden_layer_sizes;
model_artifact.activation_name = activation_name;
model_artifact.dropout_probability = dropout_probability;
model_artifact.training_config = struct( ...
    'max_epochs', max_epochs, ...
    'mini_batch_size', mini_batch_size, ...
    'initial_learning_rate', initial_learning_rate, ...
    'validation_patience', validation_patience, ...
    'shared_split_file', shared_split_file, ...
    'shared_split_hash', shared_split_info.split_hash, ...
    'training_seed', training_seed);

save(fullfile(artifacts_dir, 'surrogate_baseline_model.mat'), 'model_artifact', '-v7.3');
save(fullfile(artifacts_dir, 'surrogate_baseline_predictions.mat'), ...
    'Y_train_raw', 'Y_validation_raw', 'Y_test_raw', ...
    'Y_train_pred', 'Y_validation_pred', 'Y_test_pred', ...
    'X_train_raw', 'X_validation_raw', 'X_test_raw', ...
    'freq_grid', 'evaluation', '-v7.3');

write_metrics_report(fullfile(artifacts_dir, 'metrics_report.txt'), ...
    metrics, dataset_info, num_samples, shared_split_info);

%% Generate diagnostic plots
plot_training_history(training_info, figures_dir);
plot_prediction_scatter(Y_test_raw, Y_test_pred, figures_dir, scatter_max_points);
plot_mean_error_vs_frequency(freq_grid, Y_test_raw, Y_test_pred, figures_dir);
plot_random_curve_comparisons(freq_grid, Y_test_raw, Y_test_pred, figures_dir, num_random_curve_plots, shared_split_info.split_seed);
plot_worst_case_curves(freq_grid, Y_test_raw, Y_test_pred, test_curve_mae, figures_dir, num_worst_case_plots);
plot_curve_error_histogram(test_curve_mae, figures_dir);

fprintf('\nTraining complete.\n');
fprintf('Train MAE: %.6f | Validation MAE: %.6f | Test MAE: %.6f\n', ...
    metrics.train.mae, metrics.validation.mae, metrics.test.mae);
fprintf('Artifacts saved to %s\n', artifacts_dir);

%% Local functions
function safe_name = sanitize_run_name(value)
    safe_name = regexprep(lower(char(value)), '[^a-z0-9]+', '_');
    safe_name = regexprep(safe_name, '^_+|_+$', '');
    if isempty(safe_name)
        safe_name = 'mlp_run';
    end
end

function output_dir = unique_run_directory(output_root, run_folder_name)
    output_dir = fullfile(output_root, run_folder_name);
    if ~exist(output_dir, 'dir')
        return;
    end

    for suffix = 2:999
        candidate = fullfile(output_root, sprintf('%s_%02d', run_folder_name, suffix));
        if ~exist(candidate, 'dir')
            output_dir = candidate;
            return;
        end
    end
    error('Could not create a unique MLP run directory under: %s', output_root);
end

function has_files = directory_contains_files(directory_path)
    if ~exist(directory_path, 'dir')
        has_files = false;
        return;
    end
    entries = dir(fullfile(directory_path, '**', '*'));
    has_files = any(~[entries.isdir]);
end

function [split_indices, split_info] = load_shared_curve_split(split_file, num_samples, dataset_file)
    split_data = jsondecode(fileread(split_file));
    required_fields = { ...
        'schema_version', 'dataset_run', 'source_curve_count', 'index_base', ...
        'split_seed', 'split_hash', 'frequency_grid_independent', ...
        'train_curve_indices', 'validation_curve_indices', 'test_curve_indices'};
    for i = 1:numel(required_fields)
        if ~isfield(split_data, required_fields{i})
            error('Shared curve split is missing required field: %s', required_fields{i});
        end
    end

    if split_data.schema_version ~= 1
        error('Unsupported shared curve split schema: %g', split_data.schema_version);
    end
    if split_data.index_base ~= 1
        error('Shared curve split indices must be 1-based for MATLAB.');
    end
    if split_data.source_curve_count ~= num_samples
        error(['Shared curve split expects %d source curves, but the dataset contains %d. ', ...
               'Regenerate the split for this curve set.'], ...
              split_data.source_curve_count, num_samples);
    end
    if ~split_data.frequency_grid_independent
        error('Shared curve split must be frequency-grid independent.');
    end

    split_indices = struct();
    split_indices.train = double(split_data.train_curve_indices(:));
    split_indices.validation = double(split_data.validation_curve_indices(:));
    split_indices.test = double(split_data.test_curve_indices(:));

    combined = [split_indices.train; split_indices.validation; split_indices.test];
    if numel(combined) ~= num_samples || ...
            ~isequal(sort(combined), (1:num_samples).') || ...
            numel(unique(combined)) ~= num_samples
        error(['Shared split indices must be disjoint and cover every source curve ', ...
               'from 1 to %d exactly once.'], num_samples);
    end

    normalized_dataset_path = strrep(char(dataset_file), '\', '/');
    expected_run_token = ['/datasets/' char(split_data.dataset_run) '/'];
    if ~contains(lower(normalized_dataset_path), lower(expected_run_token))
        error('Dataset file does not match shared split dataset_run %s: %s', ...
              split_data.dataset_run, dataset_file);
    end

    split_info = struct();
    split_info.split_file = char(split_file);
    split_info.split_hash = char(split_data.split_hash);
    split_info.dataset_run = char(split_data.dataset_run);
    split_info.schema_version = split_data.schema_version;
    split_info.source_curve_count = split_data.source_curve_count;
    split_info.split_seed = split_data.split_seed;
    split_info.frequency_grid_independent = logical(split_data.frequency_grid_independent);
end

function [X_out, preprocessing] = fit_transform_inputs(X_in, apply_log10_to_inputs, log10_feature_indices, standardize_inputs)
    X_work = X_in;

    preprocessing = struct();
    preprocessing.apply_log10_to_inputs = apply_log10_to_inputs;
    preprocessing.log10_feature_indices = log10_feature_indices(:).';
    preprocessing.standardize_inputs = standardize_inputs;
    preprocessing.input_means = [];
    preprocessing.input_scales = [];

    if apply_log10_to_inputs
        non_positive_mask = X_work(:, log10_feature_indices) <= 0;
        if any(non_positive_mask(:))
            error('log10 transform requested, but at least one selected input value is non-positive.');
        end
        X_work(:, log10_feature_indices) = log10(X_work(:, log10_feature_indices));
    end

    if standardize_inputs
        [X_out, stats] = fit_standardization(X_work, true);
        preprocessing.input_means = stats.means;
        preprocessing.input_scales = stats.scales;
    else
        X_out = X_work;
    end
end

function X_out = transform_inputs(X_in, preprocessing)
    X_work = X_in;

    if preprocessing.apply_log10_to_inputs
        non_positive_mask = X_work(:, preprocessing.log10_feature_indices) <= 0;
        if any(non_positive_mask(:))
            error('log10 transform requested, but at least one selected input value is non-positive.');
        end
        X_work(:, preprocessing.log10_feature_indices) = log10(X_work(:, preprocessing.log10_feature_indices));
    end

    if preprocessing.standardize_inputs
        X_out = (X_work - preprocessing.input_means) ./ preprocessing.input_scales;
    else
        X_out = X_work;
    end
end

function [X_out, stats] = fit_standardization(X_in, do_standardize)
    stats = struct();

    if do_standardize
        stats.means = mean(X_in, 1);
        stats.scales = std(X_in, 0, 1);
        stats.scales(stats.scales < eps) = 1;
        X_out = (X_in - stats.means) ./ stats.scales;
    else
        stats.means = zeros(1, size(X_in, 2));
        stats.scales = ones(1, size(X_in, 2));
        X_out = X_in;
    end
end

function X_out = apply_standardization(X_in, stats)
    X_out = (X_in - stats.means) ./ stats.scales;
end

function X_out = invert_output_standardization(X_in, stats)
    X_out = X_in .* stats.scales + stats.means;
end

function layers = build_mlp_layers(num_features, num_outputs, hidden_layer_sizes, activation_name, dropout_probability)
    layers = [
        featureInputLayer(num_features, 'Normalization', 'none', 'Name', 'input')
    ];

    for i = 1:numel(hidden_layer_sizes)
        layers = [
            layers
            fullyConnectedLayer(hidden_layer_sizes(i), 'Name', sprintf('fc_%d', i))
            build_activation_layer(activation_name, i)
        ];

        if dropout_probability > 0
            layers = [
                layers
                dropoutLayer(dropout_probability, 'Name', sprintf('dropout_%d', i))
            ];
        end
    end

    layers = [
        layers
        fullyConnectedLayer(num_outputs, 'Name', 'output_fc')
        regressionLayer('Name', 'regression_output')
    ];
end

function layer = build_activation_layer(activation_name, layer_idx)
    switch lower(activation_name)
        case 'relu'
            layer = reluLayer('Name', sprintf('relu_%d', layer_idx));
        case 'tanh'
            layer = tanhLayer('Name', sprintf('tanh_%d', layer_idx));
        otherwise
            error('Unsupported activation_name: %s', activation_name);
    end
end

function metrics = compute_regression_metrics(y_true, y_pred)
    residual = y_pred - y_true;

    metrics = struct();
    metrics.mse = mean(residual.^2, 'all');
    metrics.rmse = sqrt(metrics.mse);
    metrics.mae = mean(abs(residual), 'all');

    denominator = sum((y_true - mean(y_true, 1)).^2, 'all');
    numerator = sum((y_true - y_pred).^2, 'all');

    if denominator < eps
        metrics.r2 = NaN;
    else
        metrics.r2 = 1 - numerator / denominator;
    end
end

function write_metrics_report(report_file, metrics, dataset_info, num_samples, shared_split_info)
    fid = fopen(report_file, 'w');
    if fid == -1
        error('Could not open metrics report for writing: %s', report_file);
    end

    cleanup_obj = onCleanup(@() fclose(fid));

    fprintf(fid, 'Surrogate Baseline Metrics Report\n');
    fprintf(fid, 'Fiber: %s\n', dataset_info.fiberfolder);
    fprintf(fid, 'Total Samples: %d\n', num_samples);
    fprintf(fid, 'Frequency Range: %.2f Hz to %.2f Hz\n', dataset_info.freq_min, dataset_info.freq_max);
    fprintf(fid, 'Frequency Points: %d\n', dataset_info.n_freq);
    fprintf(fid, 'Dataset Run: %s\n', shared_split_info.dataset_run);
    fprintf(fid, 'Shared Split File: %s\n', shared_split_info.split_file);
    fprintf(fid, 'Shared Split Hash: %s\n\n', shared_split_info.split_hash);

    fprintf(fid, 'Train Metrics\n');
    fprintf(fid, 'MSE  : %.8e\n', metrics.train.mse);
    fprintf(fid, 'RMSE : %.8e\n', metrics.train.rmse);
    fprintf(fid, 'MAE  : %.8e\n', metrics.train.mae);
    fprintf(fid, 'R2   : %.8f\n\n', metrics.train.r2);

    fprintf(fid, 'Validation Metrics\n');
    fprintf(fid, 'MSE  : %.8e\n', metrics.validation.mse);
    fprintf(fid, 'RMSE : %.8e\n', metrics.validation.rmse);
    fprintf(fid, 'MAE  : %.8e\n', metrics.validation.mae);
    fprintf(fid, 'R2   : %.8f\n\n', metrics.validation.r2);

    fprintf(fid, 'Test Metrics\n');
    fprintf(fid, 'MSE  : %.8e\n', metrics.test.mse);
    fprintf(fid, 'RMSE : %.8e\n', metrics.test.rmse);
    fprintf(fid, 'MAE  : %.8e\n', metrics.test.mae);
    fprintf(fid, 'R2   : %.8f\n', metrics.test.r2);
end

function plot_training_history(training_info, figures_dir)
    fig = figure('Visible', 'off', 'Color', 'w', 'Theme', 'light');
    hold on;

    if isfield(training_info, 'TrainingLoss') && ~isempty(training_info.TrainingLoss)
        plot(training_info.TrainingLoss, 'LineWidth', 1.5, 'DisplayName', 'Training Loss');
    end

    if isfield(training_info, 'ValidationLoss') && ~isempty(training_info.ValidationLoss)
        validation_loss = training_info.ValidationLoss;
        valid_idx = ~isnan(validation_loss);
        plot(find(valid_idx), validation_loss(valid_idx), 'LineWidth', 1.5, 'DisplayName', 'Validation Loss');
    end

    xlabel('Iteration');
    ylabel('Loss');
    title('Training History');
    legend('Location', 'northeast');
    grid on;

    save_figure(fig, fullfile(figures_dir, 'training_history.png'));
    close(fig);
end

function plot_prediction_scatter(y_true, y_pred, figures_dir, scatter_max_points)
    y_true_vec = y_true(:);
    y_pred_vec = y_pred(:);
    n_points = numel(y_true_vec);

    if n_points > scatter_max_points
        rng(777);
        idx = randperm(n_points, scatter_max_points);
        y_true_vec = y_true_vec(idx);
        y_pred_vec = y_pred_vec(idx);
    end

    fig = figure('Visible', 'off', 'Color', 'w', 'Theme', 'light');
    scatter(y_true_vec, y_pred_vec, 12, 'filled', 'MarkerFaceAlpha', 0.35);
    hold on;

    min_val = min([y_true_vec; y_pred_vec]);
    max_val = max([y_true_vec; y_pred_vec]);
    plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.25);

    xlabel('True Absorption');
    ylabel('Predicted Absorption');
    title('Predicted vs True Scatter');
    axis tight;
    grid on;

    save_figure(fig, fullfile(figures_dir, 'predicted_vs_true_scatter.png'));
    close(fig);
end

function plot_mean_error_vs_frequency(freq_grid, y_true, y_pred, figures_dir)
    mae_by_frequency = mean(abs(y_pred - y_true), 1);
    rmse_by_frequency = sqrt(mean((y_pred - y_true).^2, 1));

    fig = figure('Visible', 'off', 'Color', 'w', 'Theme', 'light');
    plot(freq_grid, mae_by_frequency, 'LineWidth', 1.75, 'DisplayName', 'MAE');
    hold on;
    plot(freq_grid, rmse_by_frequency, 'LineWidth', 1.75, 'DisplayName', 'RMSE');
    xlabel('Frequency (Hz)');
    ylabel('Error');
    title('Error vs Frequency');
    legend('Location', 'northeast');
    grid on;

    save_figure(fig, fullfile(figures_dir, 'error_vs_frequency.png'));
    close(fig);
end

function plot_random_curve_comparisons(freq_grid, y_true, y_pred, figures_dir, num_random_curve_plots, random_seed)
    n_test = size(y_true, 1);
    num_to_plot = min(num_random_curve_plots, n_test);

    rng(random_seed + 1000);
    selected_idx = randperm(n_test, num_to_plot);

    fig = figure('Visible', 'off', 'Color', 'w', 'Theme', 'light');
    tiledlayout(num_to_plot, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:num_to_plot
        nexttile;
        plot(freq_grid, y_true(selected_idx(i), :), 'k-', 'LineWidth', 1.5, 'DisplayName', 'True');
        hold on;
        plot(freq_grid, y_pred(selected_idx(i), :), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
        ylabel('\alpha');
        title(sprintf('Random Test Curve %d', selected_idx(i)));
        grid on;
        if i == 1
            legend('Location', 'best');
        end
    end

    xlabel('Frequency (Hz)');
    save_figure(fig, fullfile(figures_dir, 'random_curve_comparisons.png'));
    close(fig);
end

function plot_worst_case_curves(freq_grid, y_true, y_pred, curve_mae, figures_dir, num_worst_case_plots)
    [~, sorted_idx] = sort(curve_mae, 'descend');
    num_to_plot = min(num_worst_case_plots, numel(sorted_idx));

    fig = figure('Visible', 'off', 'Color', 'w', 'Theme', 'light');
    tiledlayout(num_to_plot, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:num_to_plot
        idx = sorted_idx(i);
        nexttile;
        plot(freq_grid, y_true(idx, :), 'k-', 'LineWidth', 1.5, 'DisplayName', 'True');
        hold on;
        plot(freq_grid, y_pred(idx, :), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
        ylabel('\alpha');
        title(sprintf('Worst-Case Curve %d | Curve MAE = %.4e', idx, curve_mae(idx)));
        grid on;
        if i == 1
            legend('Location', 'best');
        end
    end

    xlabel('Frequency (Hz)');
    save_figure(fig, fullfile(figures_dir, 'worst_case_curves.png'));
    close(fig);
end

function plot_curve_error_histogram(curve_mae, figures_dir)
    fig = figure('Visible', 'off', 'Color', 'w', 'Theme', 'light');
    histogram(curve_mae, 30);
    xlabel('Per-Curve MAE');
    ylabel('Count');
    title('Distribution of Test Curve Errors');
    grid on;

    save_figure(fig, fullfile(figures_dir, 'curve_error_histogram.png'));
    close(fig);
end

function save_figure(fig, output_file)
    if exist('exportgraphics', 'file') == 2
        exportgraphics(fig, output_file, 'Resolution', 200);
    else
        saveas(fig, output_file);
    end
end
