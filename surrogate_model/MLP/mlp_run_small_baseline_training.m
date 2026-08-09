%% Small Baseline Training Driver
% This driver trains the surrogate baseline on the small dataset for a quick
% end-to-end pipeline validation.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);

surrogate_training_config = struct();
surrogate_training_config.dataset_file = fullfile(surrogate_root, 'datasets', 'smoke', 'MLP', 'Wool_surrogate_dataset_small.mat');
surrogate_training_config.shared_split_file = fullfile(surrogate_root, 'datasets', 'smoke', 'shared_curve_split.json');
surrogate_training_config.experiment_name = 'wool_baseline_mlp_small';
surrogate_training_config.max_epochs = 80;
surrogate_training_config.mini_batch_size = 16;
surrogate_training_config.validation_patience = 10;
surrogate_training_config.hidden_layer_sizes = [64, 64];
surrogate_training_config.num_random_curve_plots = 4;
surrogate_training_config.num_worst_case_plots = 4;
surrogate_training_config.scatter_max_points = 1500;

run(fullfile(script_dir, 'mlp_train_surrogate_baseline.m'));
