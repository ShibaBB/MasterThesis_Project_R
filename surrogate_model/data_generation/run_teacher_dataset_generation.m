%% Active Dataset-Run Teacher Generation Driver
% The active run and all generation settings come from the central run config.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
addpath(surrogate_root);
run_config = resolve_dataset_run_config();

surrogate_dataset_config = struct();
surrogate_dataset_config.dataset_run = run_config.run_id;

fprintf('Generating teacher dataset for %s using %s\n', ...
    char(string(run_config.run_id)), run_config.config_file);
run(fullfile(script_dir, 'generate_teacher_dataset.m'));
