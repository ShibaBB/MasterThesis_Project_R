%% Global Symbolic Dataset Generation Driver
% This driver creates the one-segment scalar symbolic dataset for the
% planned global symbolic-regression model.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
run_config = global_dataset_run_config();

symbolic_dataset_config = struct();
symbolic_dataset_config.dataset_run = run_config.dataset_run;
symbolic_dataset_config.teacher_dataset_file = run_config.teacher_dataset_file;
symbolic_dataset_config.output_file = run_config.global_dataset_file;
symbolic_dataset_config.segment_bounds_hz = run_config.global_bounds_hz;
symbolic_dataset_config.segment_names = {run_config.global_name};

run(fullfile(surrogate_root, 'segmented_symbolic_regression', 'generate_segmented_symbolic_dataset.m'));
