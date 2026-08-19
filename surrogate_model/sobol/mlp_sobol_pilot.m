function mlp_sobol_pilot(config)
%MLP_SOBOL_PILOT Frequency-resolved Sobol screening with formal MLP models.

arguments
    config struct
end

required_files = {
    config.dataset_file, config.dataset_config_file, config.shared_split_file, ...
    config.re_model_file, config.im_model_file};
for i = 1:numel(required_files)
    if ~exist(required_files{i}, 'file')
        error('Required Sobol input file not found: %s', required_files{i});
    end
end
if exist(config.output_dir, 'dir') && directory_contains_files(config.output_dir)
    error('Refusing to overwrite an existing Sobol artifact directory: %s', config.output_dir);
end

figures_dir = fullfile(config.output_dir, 'figures');

dataset = load(config.dataset_file, 'X', 'freq_grid', 'dataset_info');
dataset_config = jsondecode(fileread(config.dataset_config_file));
split_data = jsondecode(fileread(config.shared_split_file));
re_loaded = load(config.re_model_file, 'model_artifact');
im_loaded = load(config.im_model_file, 'model_artifact');
models = struct('re', re_loaded.model_artifact, 'im', im_loaded.model_artifact);

validate_contract(config, dataset, split_data, models);

parameter_names = {'sigma', 'alpha_infinity', 'lambda', 'lambda_prime', 'k0_prime'};
d = numel(parameter_names);
freq = double(dataset.freq_grid(:)).';
n_freq = numel(freq);
sample_sizes = unique(double(config.base_sample_sizes(:)).', 'sorted');
n_max = sample_sizes(end);

[phi, h, lower_bounds, upper_bounds] = resolve_run1_bounds(dataset.dataset_info);
mkdir(config.output_dir);
mkdir(figures_dir);
write_bounds_table(config.output_dir, parameter_names, lower_bounds, upper_bounds, phi, h);

rng(config.bootstrap_seed, 'twister');
sobol_points = sobolset(2 * d, 'Skip', config.sobol_skip);
sobol_points = scramble(sobol_points, config.scramble_method);
unit_samples = net(sobol_points, n_max);
a_raw = lower_bounds + unit_samples(:, 1:d) .* (upper_bounds - lower_bounds);
b_raw = lower_bounds + unit_samples(:, d + 1:end) .* (upper_bounds - lower_bounds);

fprintf('MLP Sobol pilot: D=%d, N=%d, frequencies=%d\n', d, n_max, n_freq);
fprintf('Predicting A and B matrices for both targets...\n');

target_names = {'re', 'im'};
target_labels = {'R_real', 'R_imag'};
results = struct();
for t = 1:numel(target_names)
    target = target_names{t};
    artifact = models.(target);
    y_a = predict_mlp_curves(artifact, assemble_inputs(a_raw, phi, h), config.predict_batch_size);
    y_b = predict_mlp_curves(artifact, assemble_inputs(b_raw, phi, h), config.predict_batch_size);

    squared_total = zeros(n_max, n_freq, d);
    squared_first = zeros(n_max, n_freq, d);
    for j = 1:d
        fprintf('Target %s: hybrid matrix %d/%d (%s)\n', target, j, d, parameter_names{j});
        ab_raw = a_raw;
        ab_raw(:, j) = b_raw(:, j);
        y_ab = predict_mlp_curves(artifact, assemble_inputs(ab_raw, phi, h), config.predict_batch_size);
        squared_total(:, :, j) = (y_a - y_ab).^2;
        squared_first(:, :, j) = (y_b - y_ab).^2;
    end

    convergence = repmat(struct('n', [], 's1', [], 'st', [], 'variance', []), numel(sample_sizes), 1);
    for n_idx = 1:numel(sample_sizes)
        n = sample_sizes(n_idx);
        [s1, st, output_variance] = estimate_jansen( ...
            y_a(1:n, :), y_b(1:n, :), ...
            squared_first(1:n, :, :), squared_total(1:n, :, :));
        convergence(n_idx).n = n;
        convergence(n_idx).s1 = s1;
        convergence(n_idx).st = st;
        convergence(n_idx).variance = output_variance;
    end

    final = convergence(end);
    [s1_low, s1_high, st_low, st_high] = bootstrap_intervals( ...
        y_a, y_b, squared_first, squared_total, ...
        config.bootstrap_replicates, config.bootstrap_seed + t);

    results.(target) = struct( ...
        's1', final.s1, 'st', final.st, 'interaction_gap', final.st - final.s1, ...
        'output_variance', final.variance, ...
        's1_ci_low', s1_low, 's1_ci_high', s1_high, ...
        'st_ci_low', st_low, 'st_ci_high', st_high, ...
        'convergence', convergence, ...
        'squared_first', squared_first, 'squared_total', squared_total);

    plot_heatmap(freq, final.s1, parameter_names, ...
        sprintf('%s MLP Sobol first-order index (S1)', target_labels{t}), ...
        fullfile(figures_dir, sprintf('%s_s1_heatmap.png', target)));
    plot_heatmap(freq, final.st, parameter_names, ...
        sprintf('%s MLP Sobol total-effect index (ST)', target_labels{t}), ...
        fullfile(figures_dir, sprintf('%s_st_heatmap.png', target)));
    plot_heatmap(freq, final.st - final.s1, parameter_names, ...
        sprintf('%s MLP Sobol interaction gap (ST-S1)', target_labels{t}), ...
        fullfile(figures_dir, sprintf('%s_interaction_gap_heatmap.png', target)));
end

segments = dataset_config.segmented_sr.segments;
long_table = build_long_table(results, target_names, target_labels, freq, parameter_names, n_max);
writetable(long_table, fullfile(config.output_dir, 'frequency_sensitivity.csv'));

segment_table = build_functional_summary( ...
    results, target_names, target_labels, freq, parameter_names, segments);
writetable(segment_table, fullfile(config.output_dir, 'segment_functional_summary.csv'));

global_table = build_global_summary(results, target_names, target_labels, parameter_names);
writetable(global_table, fullfile(config.output_dir, 'global_functional_summary.csv'));

convergence_table = build_convergence_table(results, target_names, target_labels, parameter_names);
writetable(convergence_table, fullfile(config.output_dir, 'convergence_summary.csv'));

metadata = struct();
metadata.schema_version = 1;
metadata.created_local = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
metadata.timezone = 'Europe/Berlin';
metadata.method = 'Jansen first-order and total-effect indices';
metadata.emulator_family = 'MLP';
metadata.dataset_run = config.dataset_run;
metadata.dataset_file = config.dataset_file;
metadata.shared_split_file = config.shared_split_file;
metadata.shared_split_hash = char(split_data.split_hash);
metadata.model_files = struct('re', config.re_model_file, 'im', config.im_model_file);
metadata.fixed_inputs = struct('phi', phi, 'h_m', h);
metadata.parameter_names = parameter_names;
metadata.lower_bounds = lower_bounds;
metadata.upper_bounds = upper_bounds;
metadata.input_distribution = 'independent linear-uniform in raw physical units';
metadata.base_sample_sizes = sample_sizes;
metadata.sobol_skip = config.sobol_skip;
metadata.scramble_method = config.scramble_method;
metadata.bootstrap_replicates = config.bootstrap_replicates;
metadata.bootstrap_seed = config.bootstrap_seed;
metadata.frequency_points = n_freq;
metadata.frequency_min_hz = min(freq);
metadata.frequency_max_hz = max(freq);
metadata.output_postprocessing = 'none';
metadata.interpretation = ['MLP-based screening only. Confirm small effects, interactions, ', ...
    'and sensitive frequency regions against the JCAL teacher.'];
write_json(fullfile(config.output_dir, 'analysis_metadata.json'), metadata);

for t = 1:numel(target_names)
    results.(target_names{t}) = rmfield(results.(target_names{t}), ...
        {'squared_first', 'squared_total'});
end
save(fullfile(config.output_dir, 'sobol_results.mat'), ...
    'results', 'freq', 'parameter_names', 'lower_bounds', 'upper_bounds', ...
    'phi', 'h', 'sample_sizes', 'metadata', '-v7.3');
write_report(fullfile(config.output_dir, 'analysis_report.txt'), ...
    metadata, global_table, segment_table, convergence_table);

fprintf('MLP Sobol pilot complete. Artifacts saved to: %s\n', config.output_dir);
end

function validate_contract(config, dataset, split_data, models)
if ~strcmp(char(string(dataset.dataset_info.dataset_run)), config.dataset_run)
    error('Sobol dataset run mismatch.');
end
if ~strcmp(char(string(split_data.dataset_run)), config.dataset_run)
    error('Sobol shared split run mismatch.');
end
if ~strcmp(char(string(dataset.dataset_info.complex_source)), 'Reflect')
    error('Sobol dataset is not based on the complex Reflect target.');
end
for target = {'re', 'im'}
    artifact = models.(target{1});
    if ~strcmp(artifact.dataset_run, config.dataset_run) || ~strcmp(artifact.target, target{1})
        error('MLP artifact target/run mismatch for %s.', target{1});
    end
    if ~strcmp(artifact.shared_split_info.split_hash, char(split_data.split_hash))
        error('MLP artifact split hash mismatch for %s.', target{1});
    end
end
end

function [phi, h, lower_bounds, upper_bounds] = resolve_run1_bounds(dataset_info)
porosity_cases = cellstr(string(dataset_info.selected_porosityfolders));
if numel(porosity_cases) ~= 1
    error('This pilot requires one fixed porosity case; found %d.', numel(porosity_cases));
end
porosity_folder = porosity_cases{1};
phi = str2double(porosity_folder) / 100;
[thickness, ~, pressure, ~, density_humid_air, ~, ~, eta, gamma, c, ~, pr] = ...
    getFluidProperties(char(string(dataset_info.fiberfolder)), porosity_folder);
h = thickness * 1e-3;
air = struct('density_humid_air', density_humid_air, 'speed_of_sound', c, ...
    'impedance', density_humid_air * c, 'eta', eta, 'gamma', gamma, ...
    'Pr', pr, 'pressure', pressure);
[lb, ub] = getFiberConstraints(char(string(dataset_info.fiberfolder)), phi, air);
lower_bounds = double(lb(1:5));
upper_bounds = double(ub(1:5));
end

function x = assemble_inputs(material, phi, h)
n = size(material, 1);
x = [repmat(phi, n, 1), repmat(h, n, 1), material];
end

function y = predict_mlp_curves(artifact, x_raw, batch_size)
n = size(x_raw, 1);
n_outputs = numel(artifact.preprocessing.output.means);
y = zeros(n, n_outputs);
input_spec = artifact.preprocessing.input;
output_spec = artifact.preprocessing.output;
for row_start = 1:batch_size:n
    rows = row_start:min(n, row_start + batch_size - 1);
    x = x_raw(rows, :);
    if input_spec.apply_log10_to_inputs
        indices = input_spec.log10_feature_indices;
        if any(x(:, indices) <= 0, 'all')
            error('Sobol samples contain non-positive values required by MLP log10 preprocessing.');
        end
        x(:, indices) = log10(x(:, indices));
    end
    if input_spec.standardize_inputs
        x = (x - input_spec.input_means) ./ input_spec.input_scales;
    end
    y_processed = predict(artifact.net, x);
    y(rows, :) = double(y_processed) .* output_spec.scales + output_spec.means;
end
end

function [s1, st, output_variance] = estimate_jansen(y_a, y_b, squared_first, squared_total)
output_variance = var([y_a; y_b], 1, 1);
safe_variance = output_variance;
safe_variance(safe_variance < eps) = NaN;
s1 = 1 - squeeze(mean(squared_first, 1)) ./ (2 * safe_variance(:));
st = squeeze(mean(squared_total, 1)) ./ (2 * safe_variance(:));
end

function [s1_low, s1_high, st_low, st_high] = bootstrap_intervals( ...
        y_a, y_b, squared_first, squared_total, replicates, seed)
n = size(y_a, 1);
n_freq = size(y_a, 2);
d = size(squared_first, 3);
s1_boot = zeros(n_freq, d, replicates);
st_boot = zeros(n_freq, d, replicates);
rng(seed, 'twister');
for b = 1:replicates
    rows = randi(n, n, 1);
    [s1_boot(:, :, b), st_boot(:, :, b)] = estimate_jansen( ...
        y_a(rows, :), y_b(rows, :), ...
        squared_first(rows, :, :), squared_total(rows, :, :));
end
s1_low = prctile(s1_boot, 2.5, 3);
s1_high = prctile(s1_boot, 97.5, 3);
st_low = prctile(st_boot, 2.5, 3);
st_high = prctile(st_boot, 97.5, 3);
end

function table_out = build_long_table(results, targets, labels, freq, parameters, n)
rows = {};
for t = 1:numel(targets)
    r = results.(targets{t});
    for f_idx = 1:numel(freq)
        for j = 1:numel(parameters)
            rows(end + 1, :) = {targets{t}, labels{t}, freq(f_idx), parameters{j}, n, ... %#ok<AGROW>
                r.s1(f_idx, j), r.st(f_idx, j), r.interaction_gap(f_idx, j), ...
                r.s1_ci_low(f_idx, j), r.s1_ci_high(f_idx, j), ...
                r.st_ci_low(f_idx, j), r.st_ci_high(f_idx, j), r.output_variance(f_idx)};
        end
    end
end
table_out = cell2table(rows, 'VariableNames', { ...
    'target', 'target_name', 'frequency_hz', 'parameter', 'base_sample_size', ...
    'S1', 'ST', 'ST_minus_S1', 'S1_ci_low', 'S1_ci_high', ...
    'ST_ci_low', 'ST_ci_high', 'output_variance'});
end

function table_out = build_functional_summary(results, targets, labels, freq, parameters, segments)
rows = {};
for t = 1:numel(targets)
    r = results.(targets{t});
    for s = 1:numel(segments)
        bounds = double(segments(s).bounds_hz(:));
        mask = freq >= bounds(1) & freq <= bounds(2);
        for j = 1:numel(parameters)
            [s1_functional, st_functional] = functional_indices( ...
                r.squared_first(:, mask, j), r.squared_total(:, mask, j), r.output_variance(mask));
            [peak_st, local_peak] = max(r.st(mask, j));
            masked_freq = freq(mask);
            rows(end + 1, :) = {targets{t}, labels{t}, char(string(segments(s).name)), ... %#ok<AGROW>
                bounds(1), bounds(2), parameters{j}, s1_functional, st_functional, ...
                st_functional - s1_functional, mean(r.s1(mask, j), 'omitnan'), ...
                mean(r.st(mask, j), 'omitnan'), peak_st, masked_freq(local_peak)};
        end
    end
end
table_out = cell2table(rows, 'VariableNames', { ...
    'target', 'target_name', 'segment', 'lower_hz', 'upper_hz', 'parameter', ...
    'functional_S1', 'functional_ST', 'functional_ST_minus_S1', ...
    'mean_pointwise_S1', 'mean_pointwise_ST', 'max_pointwise_ST', 'peak_ST_frequency_hz'});
end

function table_out = build_global_summary(results, targets, labels, parameters)
rows = {};
for t = 1:numel(targets)
    r = results.(targets{t});
    for j = 1:numel(parameters)
        [s1_functional, st_functional] = functional_indices( ...
            r.squared_first(:, :, j), r.squared_total(:, :, j), r.output_variance);
        rows(end + 1, :) = {targets{t}, labels{t}, parameters{j}, ... %#ok<AGROW>
            s1_functional, st_functional, st_functional - s1_functional, ...
            mean(r.s1(:, j), 'omitnan'), mean(r.st(:, j), 'omitnan'), ...
            max(r.st(:, j), [], 'omitnan')};
    end
end
table_out = cell2table(rows, 'VariableNames', { ...
    'target', 'target_name', 'parameter', 'functional_S1', 'functional_ST', ...
    'functional_ST_minus_S1', 'mean_pointwise_S1', 'mean_pointwise_ST', 'max_pointwise_ST'});
end

function [s1, st] = functional_indices(squared_first, squared_total, output_variance)
denominator = 2 * sum(output_variance, 'omitnan');
s1 = 1 - sum(mean(squared_first, 1), 'all', 'omitnan') / denominator;
st = sum(mean(squared_total, 1), 'all', 'omitnan') / denominator;
end

function table_out = build_convergence_table(results, targets, labels, parameters)
rows = {};
for t = 1:numel(targets)
    convergence = results.(targets{t}).convergence;
    for n_idx = 1:numel(convergence)
        previous = [];
        if n_idx > 1
            previous = convergence(n_idx - 1);
        end
        for j = 1:numel(parameters)
            delta_s1 = NaN;
            delta_st = NaN;
            if ~isempty(previous)
                delta_s1 = max(abs(convergence(n_idx).s1(:, j) - previous.s1(:, j)), [], 'omitnan');
                delta_st = max(abs(convergence(n_idx).st(:, j) - previous.st(:, j)), [], 'omitnan');
            end
            rows(end + 1, :) = {targets{t}, labels{t}, parameters{j}, convergence(n_idx).n, ... %#ok<AGROW>
                mean(convergence(n_idx).s1(:, j), 'omitnan'), ...
                mean(convergence(n_idx).st(:, j), 'omitnan'), delta_s1, delta_st};
        end
    end
end
table_out = cell2table(rows, 'VariableNames', { ...
    'target', 'target_name', 'parameter', 'base_sample_size', ...
    'mean_pointwise_S1', 'mean_pointwise_ST', ...
    'max_abs_S1_change_from_previous_N', 'max_abs_ST_change_from_previous_N'});
end

function plot_heatmap(freq, values, parameter_names, plot_title, output_file)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1200, 500]);
imagesc(freq, 1:numel(parameter_names), values.');
axis xy;
colormap(parula);
ax = gca;
ax.Color = 'w';
ax.XColor = 'k';
ax.YColor = 'k';
ax.TickLabelInterpreter = 'none';
cb = colorbar;
cb.Color = 'k';
yticks(1:numel(parameter_names));
yticklabels(parameter_names);
xlabel('Frequency (Hz)', 'Color', 'k');
title(plot_title, 'Interpreter', 'none', 'Color', 'k');
exportgraphics(fig, output_file, 'Resolution', 200);
close(fig);
end

function write_bounds_table(output_dir, names, lower, upper, phi, h)
table_out = table(string(names(:)), lower(:), upper(:), upper(:) - lower(:), ...
    'VariableNames', {'parameter', 'lower_bound', 'upper_bound', 'span'});
writetable(table_out, fullfile(output_dir, 'parameter_bounds.csv'));
fixed = table(["phi"; "h"], [phi; h], ["dimensionless"; "m"], ...
    'VariableNames', {'parameter', 'fixed_value', 'unit'});
writetable(fixed, fullfile(output_dir, 'fixed_inputs.csv'));
end

function write_json(path, value)
fid = fopen(path, 'w');
if fid < 0
    error('Could not open JSON output: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(value, 'PrettyPrint', true));
end

function write_report(path, metadata, global_table, segment_table, convergence_table)
fid = fopen(path, 'w');
if fid < 0
    error('Could not open Sobol report: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'MLP Frequency-Resolved Sobol Pilot\n');
fprintf(fid, 'Dataset run: %s\n', metadata.dataset_run);
fprintf(fid, 'Shared split hash: %s\n', metadata.shared_split_hash);
fprintf(fid, 'Method: %s\n', metadata.method);
fprintf(fid, 'Input distribution: %s\n', metadata.input_distribution);
fprintf(fid, 'Base sample sizes: %s\n', mat2str(metadata.base_sample_sizes));
fprintf(fid, 'Bootstrap replicates: %d\n', metadata.bootstrap_replicates);
fprintf(fid, 'Fixed phi: %.8g\nFixed h (m): %.8g\n\n', ...
    metadata.fixed_inputs.phi, metadata.fixed_inputs.h_m);
for target = ["re", "im"]
    fprintf(fid, '%s global functional indices\n', upper(target));
    subset = global_table(string(global_table.target) == target, :);
    for i = 1:height(subset)
        fprintf(fid, '%-20s S1=% .6f ST=% .6f gap=% .6f\n', ...
            subset.parameter{i}, subset.functional_S1(i), subset.functional_ST(i), ...
            subset.functional_ST_minus_S1(i));
    end
    fprintf(fid, '\n');
end
fprintf(fid, 'Segment rows: %d\n', height(segment_table));
fprintf(fid, 'Convergence rows: %d\n', height(convergence_table));
fprintf(fid, '\n%s\n', metadata.interpretation);
end

function tf = directory_contains_files(path)
listing = dir(fullfile(path, '**', '*'));
tf = any(~[listing.isdir]);
end
