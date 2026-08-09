%% Active Dataset-Run MLP Baseline Training
% Paths and run identity come from surrogate_model/dataset_run_config.json.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
addpath(surrogate_root);
run_config = resolve_dataset_run_config();

surrogate_training_config = struct();
surrogate_training_config.dataset_file = run_config.paths.teacher_dataset_file;
surrogate_training_config.shared_split_file = run_config.paths.shared_split_file;
surrogate_training_config.experiment_name = sprintf('%s_baseline', char(string(run_config.run_id)));

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

fprintf('Training MLP baseline for %s using %s\n', ...
    char(string(run_config.run_id)), run_config.config_file);
run(fullfile(script_dir, 'mlp_train_surrogate_baseline.m'));
