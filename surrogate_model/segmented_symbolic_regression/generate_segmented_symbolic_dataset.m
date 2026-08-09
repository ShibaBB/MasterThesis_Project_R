%% Segmented Symbolic Dataset Generation
% This script converts a curve-based teacher dataset into a scalar segmented
% symbolic regression dataset with frequency as an explicit input variable.

clearvars -except symbolic_dataset_config;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);

%% Configuration
run_config = segmented_dataset_run_config();
dataset_run = run_config.dataset_run;
teacher_dataset_file = run_config.teacher_dataset_file;
output_file = run_config.segmented_dataset_file;
output_dir = fileparts(output_file);
overwrite_existing = false;

segment_bounds_hz = run_config.segment_bounds_hz;
segment_names = run_config.segment_names;

if exist('symbolic_dataset_config', 'var')
    if isfield(symbolic_dataset_config, 'dataset_run')
        run_config = segmented_dataset_run_config(symbolic_dataset_config.dataset_run);
        dataset_run = run_config.dataset_run;
        teacher_dataset_file = run_config.teacher_dataset_file;
        output_file = run_config.segmented_dataset_file;
        output_dir = fileparts(output_file);
        segment_bounds_hz = run_config.segment_bounds_hz;
        segment_names = run_config.segment_names;
    end
    if isfield(symbolic_dataset_config, 'teacher_dataset_file'), teacher_dataset_file = symbolic_dataset_config.teacher_dataset_file; end
    if isfield(symbolic_dataset_config, 'output_dir'), output_dir = symbolic_dataset_config.output_dir; end
    if isfield(symbolic_dataset_config, 'output_file'), output_file = symbolic_dataset_config.output_file; end
    if isfield(symbolic_dataset_config, 'segment_bounds_hz'), segment_bounds_hz = symbolic_dataset_config.segment_bounds_hz; end
    if isfield(symbolic_dataset_config, 'segment_names'), segment_names = symbolic_dataset_config.segment_names; end
    if isfield(symbolic_dataset_config, 'overwrite_existing'), overwrite_existing = symbolic_dataset_config.overwrite_existing; end
end

if exist('symbolic_dataset_config', 'var') && isfield(symbolic_dataset_config, 'output_dir') && ...
        ~isfield(symbolic_dataset_config, 'output_file')
    [~, output_name, output_ext] = fileparts(output_file);
    output_file = fullfile(output_dir, [output_name, output_ext]);
else
    output_dir = fileparts(output_file);
end

%% Validate configuration
assert_segmented_dataset_run_paths(dataset_run, teacher_dataset_file, output_file);

if ~exist(teacher_dataset_file, 'file')
    error('Teacher dataset file not found: %s', teacher_dataset_file);
end

if size(segment_bounds_hz, 2) ~= 2
    error('segment_bounds_hz must be an N x 2 array of [lower, upper] bounds.');
end

if numel(segment_names) ~= size(segment_bounds_hz, 1)
    error('segment_names must have one entry per frequency segment.');
end
if exist(output_file, 'file') && ~overwrite_existing
    error(['Refusing to overwrite existing symbolic dataset: %s. ' ...
        'Create a new dataset run, or explicitly set overwrite_existing=true.'], output_file);
end

%% Load teacher dataset
dataset = load(teacher_dataset_file);
required_variables = {'X', 'Y', 'freq_grid', 'dataset_info', 'sample_metadata'};
for i = 1:numel(required_variables)
    if ~isfield(dataset, required_variables{i})
        error('Teacher dataset is missing required variable: %s', required_variables{i});
    end
end

X_curve = dataset.X;
Y_curve = dataset.Y;
freq_grid = dataset.freq_grid(:);
dataset_info = dataset.dataset_info;
sample_metadata = dataset.sample_metadata;

if isfield(dataset_info, 'dataset_run') && ~strcmp(char(string(dataset_info.dataset_run)), dataset_run)
    error('Teacher dataset metadata belongs to %s, not configured run %s.', ...
        char(string(dataset_info.dataset_run)), dataset_run);
end
if numel(freq_grid) ~= run_config.n_freq || min(freq_grid) ~= run_config.freq_min || ...
        max(freq_grid) ~= run_config.freq_max
    error(['Teacher frequency grid does not match %s: expected %.6g-%.6g Hz with %d points, ' ...
        'found %.6g-%.6g Hz with %d points.'], run_config.dataset_config_file, ...
        run_config.freq_min, run_config.freq_max, run_config.n_freq, ...
        min(freq_grid), max(freq_grid), numel(freq_grid));
end

if size(X_curve, 1) ~= size(Y_curve, 1)
    error('Teacher dataset X and Y must have the same number of rows.');
end

if size(Y_curve, 2) ~= numel(freq_grid)
    error('Teacher dataset Y column count must match numel(freq_grid).');
end

num_curve_samples = size(X_curve, 1);
num_freq = numel(freq_grid);
num_symbolic_samples = num_curve_samples * num_freq;

feature_names = [dataset_info.parameter_names, {'f'}];

%% Expand curve dataset into scalar symbolic samples
X_repeated = repelem(X_curve, num_freq, 1);
freq_column = repmat(freq_grid, num_curve_samples, 1);

X_symbolic = [X_repeated, freq_column];
y_symbolic = reshape(Y_curve.', [], 1);

source_curve_index = repelem((1:num_curve_samples).', num_freq, 1);
source_porosityfolder = strings(num_symbolic_samples, 1);

for i = 1:num_curve_samples
    row_mask = source_curve_index == i;
    source_porosityfolder(row_mask) = string(sample_metadata(i).porosityfolder);
end

segment_index = assign_frequency_segments(freq_column, segment_bounds_hz);

segment_definitions = struct( ...
    'name', segment_names(:), ...
    'lower_hz', num2cell(segment_bounds_hz(:, 1)), ...
    'upper_hz', num2cell(segment_bounds_hz(:, 2)));

%% Save dataset
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

symbolic_dataset_info = struct();
symbolic_dataset_info.dataset_run = dataset_run;
symbolic_dataset_info.source_teacher_dataset_file = teacher_dataset_file;
symbolic_dataset_info.fiberfolder = dataset_info.fiberfolder;
symbolic_dataset_info.selected_porosityfolders = dataset_info.selected_porosityfolders;
symbolic_dataset_info.freq_min = min(freq_grid);
symbolic_dataset_info.freq_max = max(freq_grid);
symbolic_dataset_info.n_freq = num_freq;
symbolic_dataset_info.num_curve_samples = num_curve_samples;
symbolic_dataset_info.num_symbolic_samples = num_symbolic_samples;
symbolic_dataset_info.feature_names = feature_names;
symbolic_dataset_info.target_name = dataset_info.target_name;
symbolic_dataset_info.segment_bounds_hz = segment_bounds_hz;
symbolic_dataset_info.segment_names = segment_names(:);
symbolic_dataset_info.recommended_log10_feature_indices = [3, 5, 6, 7, 8];

save(output_file, ...
    'X_symbolic', ...
    'y_symbolic', ...
    'freq_grid', ...
    'source_curve_index', ...
    'source_porosityfolder', ...
    'segment_index', ...
    'segment_definitions', ...
    'symbolic_dataset_info', ...
    '-v7.3');

fprintf('Saved symbolic dataset to %s\n', output_file);
fprintf('Curve samples: %d | Frequency points: %d | Symbolic samples: %d\n', ...
    num_curve_samples, num_freq, num_symbolic_samples);

%% Local functions
function segment_index = assign_frequency_segments(freq_values, segment_bounds_hz)
    segment_index = zeros(numel(freq_values), 1);

    for seg_idx = 1:size(segment_bounds_hz, 1)
        lower_hz = segment_bounds_hz(seg_idx, 1);
        upper_hz = segment_bounds_hz(seg_idx, 2);

        if seg_idx < size(segment_bounds_hz, 1)
            in_segment = freq_values >= lower_hz & freq_values < upper_hz;
        else
            in_segment = freq_values >= lower_hz & freq_values <= upper_hz;
        end

        segment_index(in_segment) = seg_idx;
    end

    if any(segment_index == 0)
        missing_freq = unique(freq_values(segment_index == 0));
        error('Some frequency values do not belong to any segment. First missing value: %.6f Hz', missing_freq(1));
    end
end
