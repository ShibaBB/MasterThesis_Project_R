%% Small Dataset Inspection Driver
% This driver runs inspection on the small dataset generated for quick checks.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);

surrogate_inspection_config = struct();
surrogate_inspection_config.dataset_file = fullfile(surrogate_root, 'datasets', 'smoke', 'MLP', 'Wool_surrogate_dataset_small.mat');
surrogate_inspection_config.inspection_name = 'wool_dataset_inspection_small';

run(fullfile(script_dir, 'inspect_teacher_dataset.m'));
