%% Run 2 Formal MLP Baseline Training
% Train the unchanged baseline MLP on the shared run2 teacher dataset.
% All experiment settings are stated explicitly here so the formal baseline
% remains reproducible if defaults in the underlying training script change.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);

surrogate_training_config = struct();
surrogate_training_config.dataset_file = fullfile( ...
    surrogate_root, 'datasets', 'run2', 'MLP', 'Wool_surrogate_dataset.mat');
surrogate_training_config.shared_split_file = fullfile( ...
    surrogate_root, 'datasets', 'run2', 'shared_curve_split.json');
surrogate_training_config.experiment_name = 'run2_baseline';

surrogate_training_config.apply_log10_to_inputs = true;
surrogate_training_config.log10_feature_indices = [3, 5, 6, 7];
surrogate_training_config.standardize_inputs = true;
surrogate_training_config.standardize_outputs = true;

surrogate_training_config.hidden_layer_sizes = [128, 128, 64];
surrogate_training_config.activation_name = 'relu';
surrogate_training_config.dropout_probability = 0.0;

surrogate_training_config.max_epochs = 400;
surrogate_training_config.mini_batch_size = 64;
surrogate_training_config.initial_learning_rate = 1e-3;
surrogate_training_config.validation_patience = 25;
surrogate_training_config.execution_environment = 'cpu';
surrogate_training_config.training_seed = 321;

surrogate_training_config.num_random_curve_plots = 5;
surrogate_training_config.num_worst_case_plots = 5;
surrogate_training_config.scatter_max_points = 3000;

if ~exist(surrogate_training_config.dataset_file, 'file')
    error('Run2 MLP dataset not found: %s', surrogate_training_config.dataset_file);
end

run(fullfile(script_dir, 'mlp_train_surrogate_baseline.m'));
