function config = resolve_dataset_run_config(dataset_run)
%RESOLVE_DATASET_RUN_CONFIG Load and validate the central per-run configuration.

surrogate_root = fileparts(mfilename('fullpath'));
active_file = fullfile(surrogate_root, 'dataset_run_config.json');
if ~exist(active_file, 'file')
    error('Active dataset-run configuration not found: %s', active_file);
end
active = jsondecode(fileread(active_file));
if ~isfield(active, 'schema_version') || active.schema_version ~= 1
    error('Unsupported active dataset-run configuration schema: %s', active_file);
end

if nargin < 1 || isempty(dataset_run)
    dataset_run = active.default_dataset_run;
end
dataset_run = char(string(dataset_run));
if isempty(regexp(dataset_run, '^[A-Za-z0-9][A-Za-z0-9_-]*$', 'once'))
    error('Invalid dataset run name: %s', dataset_run);
end

config_filename = 'dataset_config.json';
if isfield(active, 'run_config_filename')
    config_filename = char(string(active.run_config_filename));
end
[config_parent, config_base, config_ext] = fileparts(config_filename);
if ~isempty(config_parent) || ~strcmp([config_base, config_ext], config_filename)
    error('run_config_filename must be a plain filename.');
end

run_dir = fullfile(surrogate_root, 'datasets', dataset_run);
config_file = fullfile(run_dir, config_filename);
if ~exist(config_file, 'file')
    error('Dataset-run configuration not found: %s', config_file);
end
config = jsondecode(fileread(config_file));
if ~isfield(config, 'schema_version') || config.schema_version ~= 1
    error('Unsupported dataset-run configuration schema: %s', config_file);
end
if ~isfield(config, 'run_id') || ~strcmp(char(string(config.run_id)), dataset_run)
    error('Dataset config run_id must match its datasets/%s directory.', dataset_run);
end

required_generation = {'material', 'porosity_cases', 'sampling_method', 'random_seed', 'curve_samples', 'frequency_grid_hz'};
for i = 1:numel(required_generation)
    if ~isfield(config.generation, required_generation{i})
        error('Dataset config is missing generation.%s.', required_generation{i});
    end
end
frequency = config.generation.frequency_grid_hz;
if frequency.min >= frequency.max || frequency.points < 2 || config.generation.curve_samples < 1
    error('Dataset config contains an invalid frequency grid or sample count.');
end

if ~isfield(config, 'segmented_sr') || ~isfield(config.segmented_sr, 'segments') || ...
        ~isfield(config, 'global_sr') || ~isfield(config.global_sr, 'bounds_hz') || ...
        ~isfield(config.global_sr, 'name')
    error('Dataset config must define segmented_sr.segments and global_sr.');
end
segments = config.segmented_sr.segments;
segment_count = numel(segments);
segment_bounds_hz = zeros(segment_count, 2);
segment_names = cell(segment_count, 1);
for i = 1:segment_count
    segment_names{i} = char(string(segments(i).name));
    segment_bounds_hz(i, :) = reshape(segments(i).bounds_hz, 1, []);
end
if size(segment_bounds_hz, 2) ~= 2 || segment_bounds_hz(1, 1) ~= frequency.min || ...
        segment_bounds_hz(end, 2) ~= frequency.max || ...
        any(segment_bounds_hz(:, 1) >= segment_bounds_hz(:, 2)) || ...
        any(segment_bounds_hz(2:end, 1) ~= segment_bounds_hz(1:end-1, 2))
    error('Segment bounds must be contiguous and cover the complete configured frequency grid.');
end
global_bounds_hz = reshape(config.global_sr.bounds_hz, 1, []);
if numel(global_bounds_hz) ~= 2 || any(global_bounds_hz ~= [frequency.min, frequency.max])
    error('global_sr.bounds_hz must match the configured frequency range.');
end
config.segment_bounds_hz = segment_bounds_hz;
config.segment_names = segment_names;
config.global_bounds_hz = global_bounds_hz;
config.global_name = char(string(config.global_sr.name));
grid = linspace(frequency.min, frequency.max, frequency.points);
segment_frequency_point_counts = zeros(segment_count, 1);
for i = 1:segment_count
    if i < segment_count
        segment_frequency_point_counts(i) = sum(grid >= segment_bounds_hz(i, 1) & grid < segment_bounds_hz(i, 2));
    else
        segment_frequency_point_counts(i) = sum(grid >= segment_bounds_hz(i, 1) & grid <= segment_bounds_hz(i, 2));
    end
end
if any(segment_frequency_point_counts == 0)
    error('Every segmented SR domain must contain at least one configured frequency point.');
end
config.segment_frequency_point_counts = segment_frequency_point_counts;

material = char(string(config.generation.material));
config.config_file = config_file;
config.paths = struct();
config.paths.run_dir = run_dir;
config.paths.manifest_file = fullfile(run_dir, 'dataset_manifest.json');
config.paths.teacher_dataset_file = fullfile(run_dir, 'MLP', sprintf('%s_surrogate_dataset.mat', material));
config.paths.shared_split_file = fullfile(run_dir, 'shared_curve_split.json');
config.paths.segmented_dataset_file = fullfile(run_dir, 'segmented_SR', sprintf('%s_symbolic_segmented.mat', material));
config.paths.global_dataset_file = fullfile(run_dir, 'global_SR', sprintf('%s_symbolic_global.mat', material));
end
