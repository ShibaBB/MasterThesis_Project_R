%% Run the independent MLP-based frequency-resolved Sobol pilot.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
project_root = fileparts(surrogate_root);

sobol_config = struct();
sobol_config.dataset_run = 'run1';
sobol_config.dataset_file = fullfile( ...
    surrogate_root, 'datasets', 'run1', 'MLP', 'Wool_R.mat');
sobol_config.dataset_config_file = fullfile( ...
    surrogate_root, 'datasets', 'run1', 'dataset_config.json');
sobol_config.shared_split_file = fullfile( ...
    surrogate_root, 'datasets', 'run1', 'shared_curve_split.json');
sobol_config.re_model_file = fullfile( ...
    surrogate_root, 'MLP', 'artifacts', 're', '20260810_run1', ...
    'surrogate_baseline_model.mat');
sobol_config.im_model_file = fullfile( ...
    surrogate_root, 'MLP', 'artifacts', 'im', '20260810_run1', ...
    'surrogate_baseline_model.mat');
sobol_config.output_dir = fullfile( ...
    script_dir, 'artifacts', 'mlp', '20260810_run1_n4096');

sobol_config.base_sample_sizes = [1024, 2048, 4096];
sobol_config.sobol_skip = 1024;
sobol_config.scramble_method = 'MatousekAffineOwen';
sobol_config.predict_batch_size = 8192;
sobol_config.bootstrap_replicates = 100;
sobol_config.bootstrap_seed = 20260810;

addpath(project_root);
addpath(surrogate_root);
addpath(script_dir);
cd(project_root);

mlp_sobol_pilot(sobol_config);
