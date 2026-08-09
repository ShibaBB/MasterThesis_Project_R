function config = global_dataset_run_config(dataset_run)
%GLOBAL_DATASET_RUN_CONFIG Resolve shared dataset paths for global SR.

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
addpath(surrogate_root);
if nargin < 1
    dataset_run = '';
end
shared = resolve_dataset_run_config(dataset_run);

config = struct();
config.dataset_run = char(string(shared.run_id));
config.run_dir = shared.paths.run_dir;
config.teacher_dataset_file = shared.paths.teacher_dataset_file;
config.global_dataset_file = shared.paths.global_dataset_file;
config.global_bounds_hz = shared.global_bounds_hz;
config.global_name = shared.global_name;
config.dataset_config_file = shared.config_file;
end
