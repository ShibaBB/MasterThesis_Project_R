%% Full Segmented Symbolic Dataset Inspection Driver
% This driver inspects the default scalar segmented symbolic dataset before PySR
% training.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));

symbolic_inspection_config = struct();
run_config = segmented_dataset_run_config();
symbolic_inspection_config.dataset_run = run_config.dataset_run;
symbolic_inspection_config.dataset_file = run_config.segmented_dataset_file;
symbolic_inspection_config.inspection_name = sprintf( ...
    'wool_segmented_symbolic_dataset_inspection_%s', char(string(run_config.dataset_run)));

run(fullfile(script_dir, 'inspect_segmented_symbolic_dataset.m'));
