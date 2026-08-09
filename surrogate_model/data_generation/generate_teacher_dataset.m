%% Surrogate Dataset Generation
% This script generates paired real/imaginary reflection-coefficient targets.
% The current version is configured for Wool and starts from porosity 92,
% while keeping the workflow compatible with porosities 92 to 99.

clearvars -except surrogate_dataset_config;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
project_root = fileparts(surrogate_root);
original_dir = pwd;
cleanup_obj = onCleanup(@() cd(original_dir)); %#ok<NASGU>
cd(project_root);

%% Configuration
addpath(surrogate_root);
requested_dataset_run = '';
if exist('surrogate_dataset_config', 'var') && isfield(surrogate_dataset_config, 'dataset_run')
    requested_dataset_run = surrogate_dataset_config.dataset_run;
end
run_config = resolve_dataset_run_config(requested_dataset_run);
dataset_run = char(string(run_config.run_id));

fiberfolder = char(string(run_config.generation.material));
available_porosityfolders = {'92', '93', '94', '95', '96', '97', '98', '99'};
selected_porosityfolders = cellstr(string(run_config.generation.porosity_cases));
freq_min = run_config.generation.frequency_grid_hz.min;
freq_max = run_config.generation.frequency_grid_hz.max;
n_freq = run_config.generation.frequency_grid_hz.points;
n_samples = run_config.generation.curve_samples;
sampling_method = char(string(run_config.generation.sampling_method));
random_seed = run_config.generation.random_seed;
output_file = run_config.paths.teacher_dataset_file;
output_dir = fileparts(output_file);
allow_custom_output = false;
overwrite_existing = false;

if exist('surrogate_dataset_config', 'var')
    if isfield(surrogate_dataset_config, 'fiberfolder'), fiberfolder = surrogate_dataset_config.fiberfolder; end
    if isfield(surrogate_dataset_config, 'available_porosityfolders'), available_porosityfolders = surrogate_dataset_config.available_porosityfolders; end
    if isfield(surrogate_dataset_config, 'selected_porosityfolders'), selected_porosityfolders = surrogate_dataset_config.selected_porosityfolders; end
    if isfield(surrogate_dataset_config, 'freq_min'), freq_min = surrogate_dataset_config.freq_min; end
    if isfield(surrogate_dataset_config, 'freq_max'), freq_max = surrogate_dataset_config.freq_max; end
    if isfield(surrogate_dataset_config, 'n_freq'), n_freq = surrogate_dataset_config.n_freq; end
    if isfield(surrogate_dataset_config, 'n_samples'), n_samples = surrogate_dataset_config.n_samples; end
    if isfield(surrogate_dataset_config, 'sampling_method'), sampling_method = surrogate_dataset_config.sampling_method; end
    if isfield(surrogate_dataset_config, 'random_seed'), random_seed = surrogate_dataset_config.random_seed; end
    if isfield(surrogate_dataset_config, 'output_dir'), output_dir = surrogate_dataset_config.output_dir; end
    if isfield(surrogate_dataset_config, 'output_file'), output_file = surrogate_dataset_config.output_file; end
    if isfield(surrogate_dataset_config, 'allow_custom_output'), allow_custom_output = surrogate_dataset_config.allow_custom_output; end
    if isfield(surrogate_dataset_config, 'overwrite_existing'), overwrite_existing = surrogate_dataset_config.overwrite_existing; end
end

% Construct the filename only after all configuration overrides are resolved.
if exist('surrogate_dataset_config', 'var') && isfield(surrogate_dataset_config, 'output_dir') && ...
        ~isfield(surrogate_dataset_config, 'output_file')
    output_file = fullfile(output_dir, sprintf('%s_surrogate_dataset.mat', fiberfolder));
else
    output_dir = fileparts(output_file);
end

%% Validate configuration
if ~all(ismember(selected_porosityfolders, available_porosityfolders))
    error('selected_porosityfolders must be chosen from 92 to 99.');
end
if ~allow_custom_output
    assert_dataset_run_path(dataset_run, output_file);
    configured = run_config.generation;
    configured_porosity = cellstr(string(configured.porosity_cases));
    if ~strcmp(fiberfolder, char(string(configured.material))) || ...
            ~isequal(selected_porosityfolders(:), configured_porosity(:)) || ...
            freq_min ~= configured.frequency_grid_hz.min || ...
            freq_max ~= configured.frequency_grid_hz.max || ...
            n_freq ~= configured.frequency_grid_hz.points || ...
            n_samples ~= configured.curve_samples || ...
            ~strcmpi(sampling_method, char(string(configured.sampling_method))) || ...
            random_seed ~= configured.random_seed
        error(['Standard dataset output must use the defining values in %s. ' ...
            'Edit the per-run config instead of overriding generation identity fields.'], ...
            run_config.config_file);
    end
end

rng(random_seed);
freq_grid = linspace(freq_min, freq_max, n_freq);

if exist(output_file, 'file') && ~overwrite_existing
    error(['Refusing to overwrite existing teacher dataset: %s. ' ...
        'Create a new dataset run, or explicitly set overwrite_existing=true.'], output_file);
end

%% Prepare storage
num_porosity_cases = numel(selected_porosityfolders);
samples_per_porosity = floor(n_samples / num_porosity_cases);
remainder_samples = mod(n_samples, num_porosity_cases);
samples_per_case = samples_per_porosity * ones(num_porosity_cases, 1);
samples_per_case(1:remainder_samples) = samples_per_case(1:remainder_samples) + 1;

X = zeros(n_samples, 7);
Y_re = zeros(n_samples, n_freq);
Y_im = zeros(n_samples, n_freq);

sample_metadata = struct( ...
    'fiberfolder', cell(n_samples, 1), ...
    'porosityfolder', cell(n_samples, 1), ...
    'thickness_mm', zeros(n_samples, 1), ...
    'air_temperature_K', zeros(n_samples, 1), ...
    'pressure_Pa', zeros(n_samples, 1), ...
    'relative_humidity', zeros(n_samples, 1));

parameter_names = {'phi', 'h', 'sigma', 'alpha_infinity', 'lambda', 'lambda_prime', 'k0_prime'};
target_names = {'R_real', 'R_imag'};

%% Generate dataset
row_start = 1;

for p_idx = 1:num_porosity_cases
    porosityfolder = selected_porosityfolders{p_idx};
    phi = str2double(porosityfolder) / 100;
    n_case = samples_per_case(p_idx);

    [thickness, temperature_K, pressure, rel_humidity, density_humid_air, ~, ~, eta, gamma, c, ~, Pr] = ...
        getFluidProperties(fiberfolder, porosityfolder);

    h = thickness * 1e-3;

    airProperties = struct();
    airProperties.density_humid_air = density_humid_air;
    airProperties.speed_of_sound = c;
    airProperties.impedance = density_humid_air * c;
    airProperties.eta = eta;
    airProperties.gamma = gamma;
    airProperties.Pr = Pr;
    airProperties.pressure = pressure;

    [lb_full, ub_full] = getFiberConstraints(fiberfolder, phi, airProperties);
    lb = lb_full(1:5);
    ub = ub_full(1:5);

    param_samples = sample_parameter_space(n_case, lb, ub, sampling_method);

    row_end = row_start + n_case - 1;

    for local_idx = 1:n_case
        sigma = param_samples(local_idx, 1);
        alpha_infinity = param_samples(local_idx, 2);
        lambda = param_samples(local_idx, 3);
        lambda_prime = param_samples(local_idx, 4);
        k0_prime = param_samples(local_idx, 5);

        [Reflect, ~, ~, ~, ~, ~, ~] = jcal_reflection( ...
            h, phi, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, freq_grid, airProperties);

        global_idx = row_start + local_idx - 1;

        X(global_idx, :) = [phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime];
        Y_re(global_idx, :) = real(Reflect(:)).';
        Y_im(global_idx, :) = imag(Reflect(:)).';

        sample_metadata(global_idx).fiberfolder = fiberfolder;
        sample_metadata(global_idx).porosityfolder = porosityfolder;
        sample_metadata(global_idx).thickness_mm = thickness;
        sample_metadata(global_idx).air_temperature_K = temperature_K;
        sample_metadata(global_idx).pressure_Pa = pressure;
        sample_metadata(global_idx).relative_humidity = rel_humidity;
    end

    fprintf('Generated %d samples for %s porosity %s.\n', n_case, fiberfolder, porosityfolder);
    row_start = row_end + 1;
end

%% Save dataset
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

dataset_info = struct();
dataset_info.dataset_run = dataset_run;
dataset_info.dataset_config_file = run_config.config_file;
dataset_info.fiberfolder = fiberfolder;
dataset_info.selected_porosityfolders = selected_porosityfolders;
dataset_info.available_porosityfolders = available_porosityfolders;
dataset_info.freq_min = freq_min;
dataset_info.freq_max = freq_max;
dataset_info.n_freq = n_freq;
dataset_info.freq_grid = freq_grid;
dataset_info.n_samples = n_samples;
dataset_info.samples_per_case = samples_per_case;
dataset_info.sampling_method = sampling_method;
dataset_info.random_seed = random_seed;
dataset_info.parameter_names = parameter_names;
dataset_info.target_names = target_names;
dataset_info.complex_source = 'Reflect';
dataset_info.target_definition = struct( ...
    'R_real', 'real(Reflect)', ...
    'R_imag', 'imag(Reflect)');

if any(~isfinite(Y_re), 'all') || any(~isfinite(Y_im), 'all')
    error('Generated reflection targets contain non-finite values.');
end
save(output_file, 'X', 'Y_re', 'Y_im', 'freq_grid', 'dataset_info', 'sample_metadata', '-v7.3');

fprintf('Saved dataset to %s\n', output_file);

%% Local functions
function samples = sample_parameter_space(n_samples, lb, ub, sampling_method)
    n_dims = numel(lb);

    switch lower(sampling_method)
        case 'lhs'
            if exist('lhsdesign', 'file') == 2
                unit_samples = lhsdesign(n_samples, n_dims, 'criterion', 'maximin', 'iterations', 50);
            else
                error(['sampling_method is set to ''lhs'', but lhsdesign is not available. ', ...
                       'Install the required MATLAB functionality or switch sampling_method to ''random''.']);
            end
        case 'random'
            unit_samples = rand(n_samples, n_dims);
        otherwise
            error('Unsupported sampling_method: %s', sampling_method);
    end

    samples = lb + unit_samples .* (ub - lb);
end

function assert_dataset_run_path(dataset_run, candidate_path)
    tokens = regexp(char(string(candidate_path)), '[\\/]datasets[\\/]([^\\/]+)[\\/]', 'tokens', 'once');
    if isempty(tokens) || ~strcmp(tokens{1}, dataset_run)
        error(['Teacher output must belong to configured dataset run %s: %s. ' ...
            'Set allow_custom_output=true only for an intentional smoke/custom output.'], ...
            dataset_run, candidate_path);
    end
end
