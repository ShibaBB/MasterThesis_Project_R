%% Validate the MLP Sobol pilot against the JCAL teacher on a shared design.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
project_root = fileparts(surrogate_root);

config = struct();
config.dataset_run = 'run1';
config.dataset_file = fullfile( ...
    surrogate_root, 'datasets', 'run1', 'MLP', 'Wool_R.mat');
config.dataset_config_file = fullfile( ...
    surrogate_root, 'datasets', 'run1', 'dataset_config.json');
config.shared_split_file = fullfile( ...
    surrogate_root, 'datasets', 'run1', 'shared_curve_split.json');
config.mlp_sobol_file = fullfile( ...
    script_dir, 'artifacts', 'mlp', '20260810_run1_n4096', 'sobol_results.mat');
config.mlp_metadata_file = fullfile( ...
    script_dir, 'artifacts', 'mlp', '20260810_run1_n4096', 'analysis_metadata.json');
config.output_dir = fullfile( ...
    script_dir, 'artifacts', 'teacher', '20260810_run1_n1024');
config.base_sample_size = 1024;
config.sobol_skip = 1024;
config.scramble_method = 'MatousekAffineOwen';
config.sampling_seed = 20260810;
config.bootstrap_replicates = 100;
config.bootstrap_seed = 20260820;

addpath(project_root);
addpath(surrogate_root);
addpath(script_dir);
cd(project_root);

teacher_sobol_validation(config);
