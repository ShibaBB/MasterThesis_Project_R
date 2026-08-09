%% Small Dataset Generation Driver
% This driver runs a small dataset generation pass for quick pipeline checks.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);

surrogate_dataset_config = struct();
surrogate_dataset_config.dataset_run = 'run3';
surrogate_dataset_config.n_samples = 50;
surrogate_dataset_config.selected_porosityfolders = {'92'};
surrogate_dataset_config.output_file = fullfile(surrogate_root, 'datasets', 'smoke', 'MLP', 'Wool_surrogate_dataset_small.mat');
surrogate_dataset_config.allow_custom_output = true;

run(fullfile(script_dir, 'generate_teacher_dataset.m'));
