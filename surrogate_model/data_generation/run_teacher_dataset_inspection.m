%% Active Dataset-Run Teacher Inspection Driver

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
addpath(surrogate_root);
run_config = resolve_dataset_run_config();

surrogate_inspection_config = struct();
surrogate_inspection_config.dataset_file = run_config.paths.teacher_dataset_file;
surrogate_inspection_config.inspection_name = sprintf('%s_dataset_inspection_%s', ...
    lower(char(string(run_config.generation.material))), char(string(run_config.run_id)));

run(fullfile(script_dir, 'inspect_teacher_dataset.m'));
