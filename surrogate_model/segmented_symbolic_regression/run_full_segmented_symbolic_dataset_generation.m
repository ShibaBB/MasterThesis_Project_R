%% Full Segmented Symbolic Dataset Generation Driver
% This driver converts the default teacher dataset into the scalar segmented
% symbolic dataset used by the segment-wise PySR training workflow.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);

symbolic_dataset_config = struct();
run_config = segmented_dataset_run_config();
symbolic_dataset_config.dataset_run = run_config.dataset_run;
symbolic_dataset_config.teacher_dataset_file = run_config.teacher_dataset_file;
symbolic_dataset_config.output_file = run_config.segmented_dataset_file;

run(fullfile(script_dir, 'generate_segmented_symbolic_dataset.m'));
