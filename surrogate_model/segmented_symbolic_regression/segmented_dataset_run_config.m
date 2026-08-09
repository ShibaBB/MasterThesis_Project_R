function config = segmented_dataset_run_config(dataset_run)
%SEGMENTED_DATASET_RUN_CONFIG Resolve the shared dataset paths for segmented SR.

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
config.manifest_file = shared.paths.manifest_file;
config.teacher_dataset_file = shared.paths.teacher_dataset_file;
config.segmented_dataset_file = shared.paths.segmented_dataset_file;
config.segment_bounds_hz = shared.segment_bounds_hz;
config.segment_names = shared.segment_names;
config.freq_min = shared.generation.frequency_grid_hz.min;
config.freq_max = shared.generation.frequency_grid_hz.max;
config.n_freq = shared.generation.frequency_grid_hz.points;
config.dataset_config_file = shared.config_file;
end
