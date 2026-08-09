%% Global Symbolic Dataset Inspection Driver
% This driver inspects the one-segment scalar symbolic dataset for the
% planned global symbolic-regression model.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
run_config = global_dataset_run_config();

symbolic_inspection_config = struct();
symbolic_inspection_config.dataset_run = run_config.dataset_run;
symbolic_inspection_config.dataset_file = run_config.global_dataset_file;
symbolic_inspection_config.inspection_name = sprintf( ...
    'wool_symbolic_global_dataset_inspection_%s', char(string(run_config.dataset_run)));
symbolic_inspection_config.artifacts_dir = fullfile(script_dir, 'artifacts', symbolic_inspection_config.inspection_name);
symbolic_inspection_config.figures_dir = fullfile(symbolic_inspection_config.artifacts_dir, 'figures');

run(fullfile(surrogate_root, 'segmented_symbolic_regression', 'inspect_segmented_symbolic_dataset.m'));
