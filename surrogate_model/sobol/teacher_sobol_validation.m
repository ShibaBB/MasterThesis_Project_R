function teacher_sobol_validation(config)
%TEACHER_SOBOL_VALIDATION Compare JCAL and MLP Sobol indices on one design.

arguments
    config struct
end

required_files = {config.dataset_file, config.dataset_config_file, ...
    config.shared_split_file, config.mlp_sobol_file, config.mlp_metadata_file};
for i = 1:numel(required_files)
    if ~exist(required_files{i}, 'file')
        error('Required teacher-validation input not found: %s', required_files{i});
    end
end
if exist(config.output_dir, 'dir') && directory_contains_files(config.output_dir)
    error('Refusing to overwrite an existing teacher Sobol directory: %s', config.output_dir);
end

dataset = load(config.dataset_file, 'freq_grid', 'dataset_info');
dataset_config = jsondecode(fileread(config.dataset_config_file));
split_data = jsondecode(fileread(config.shared_split_file));
mlp_metadata = jsondecode(fileread(config.mlp_metadata_file));
mlp_saved = load(config.mlp_sobol_file, 'results', 'parameter_names', 'freq');

validate_contract(config, dataset, split_data, mlp_metadata, mlp_saved);

parameter_names = cellstr(string(mlp_metadata.parameter_names));
lower_bounds = double(mlp_metadata.lower_bounds(:)).';
upper_bounds = double(mlp_metadata.upper_bounds(:)).';
phi = double(mlp_metadata.fixed_inputs.phi);
h = double(mlp_metadata.fixed_inputs.h_m);
freq = double(dataset.freq_grid(:)).';
d = numel(parameter_names);
n = config.base_sample_size;

[air, physical_h] = resolve_air_properties(dataset.dataset_info, phi);
if abs(physical_h - h) > 1e-12
    error('Teacher-validation thickness does not match MLP Sobol metadata.');
end

rng(config.sampling_seed, 'twister');
points = sobolset(2 * d, 'Skip', config.sobol_skip);
points = scramble(points, config.scramble_method);
unit_samples = net(points, n);
a_raw = lower_bounds + unit_samples(:, 1:d) .* (upper_bounds - lower_bounds);
b_raw = lower_bounds + unit_samples(:, d + 1:end) .* (upper_bounds - lower_bounds);

fprintf('Teacher Sobol validation: D=%d, N=%d, frequencies=%d\n', d, n, numel(freq));
fprintf('Evaluating teacher matrix A...\n');
[a_re, a_im] = evaluate_teacher(a_raw, phi, h, freq, air);
fprintf('Evaluating teacher matrix B...\n');
[b_re, b_im] = evaluate_teacher(b_raw, phi, h, freq, air);

sq_first = struct('re', zeros(n, numel(freq), d), 'im', zeros(n, numel(freq), d));
sq_total = struct('re', zeros(n, numel(freq), d), 'im', zeros(n, numel(freq), d));
for j = 1:d
    fprintf('Evaluating teacher hybrid %d/%d (%s)...\n', j, d, parameter_names{j});
    ab_raw = a_raw;
    ab_raw(:, j) = b_raw(:, j);
    [ab_re, ab_im] = evaluate_teacher(ab_raw, phi, h, freq, air);
    sq_total.re(:, :, j) = (a_re - ab_re).^2;
    sq_first.re(:, :, j) = (b_re - ab_re).^2;
    sq_total.im(:, :, j) = (a_im - ab_im).^2;
    sq_first.im(:, :, j) = (b_im - ab_im).^2;
end

targets = {'re', 'im'};
target_labels = {'R_real', 'R_imag'};
y_a = struct('re', a_re, 'im', a_im);
y_b = struct('re', b_re, 'im', b_im);
results = struct();
for t = 1:numel(targets)
    target = targets{t};
    [s1, st, variance] = estimate_jansen( ...
        y_a.(target), y_b.(target), sq_first.(target), sq_total.(target));
    [s1_low, s1_high, st_low, st_high] = bootstrap_intervals( ...
        y_a.(target), y_b.(target), sq_first.(target), sq_total.(target), ...
        config.bootstrap_replicates, config.bootstrap_seed + t);
    results.(target) = struct('s1', s1, 'st', st, ...
        'interaction_gap', st - s1, 'output_variance', variance, ...
        's1_ci_low', s1_low, 's1_ci_high', s1_high, ...
        'st_ci_low', st_low, 'st_ci_high', st_high);
end

mlp_at_n = select_mlp_convergence(mlp_saved.results, n, targets);

mkdir(config.output_dir);
figures_dir = fullfile(config.output_dir, 'figures');
mkdir(figures_dir);

frequency_table = build_frequency_table( ...
    results, mlp_at_n, targets, target_labels, freq, parameter_names, n);
writetable(frequency_table, fullfile(config.output_dir, 'teacher_mlp_frequency_comparison.csv'));

agreement_table = build_agreement_table(results, mlp_at_n, targets, target_labels);
writetable(agreement_table, fullfile(config.output_dir, 'agreement_summary.csv'));

global_table = build_functional_comparison( ...
    results, mlp_at_n, targets, target_labels, parameter_names, true(1, numel(freq)), 'global');
writetable(global_table, fullfile(config.output_dir, 'global_functional_comparison.csv'));

segment_table = table();
segments = dataset_config.segmented_sr.segments;
for s = 1:numel(segments)
    bounds = double(segments(s).bounds_hz(:));
    mask = freq >= bounds(1) & freq <= bounds(2);
    part = build_functional_comparison( ...
        results, mlp_at_n, targets, target_labels, parameter_names, mask, ...
        char(string(segments(s).name)));
    part.lower_hz = repmat(bounds(1), height(part), 1);
    part.upper_hz = repmat(bounds(2), height(part), 1);
    segment_table = [segment_table; part]; %#ok<AGROW>
end
writetable(segment_table, fullfile(config.output_dir, 'segment_functional_comparison.csv'));

for t = 1:numel(targets)
    target = targets{t};
    plot_heatmap(freq, results.(target).s1, parameter_names, ...
        sprintf('%s teacher Sobol first-order index (S1)', target_labels{t}), ...
        fullfile(figures_dir, sprintf('%s_teacher_s1_heatmap.png', target)));
    plot_heatmap(freq, results.(target).st, parameter_names, ...
        sprintf('%s teacher Sobol total-effect index (ST)', target_labels{t}), ...
        fullfile(figures_dir, sprintf('%s_teacher_st_heatmap.png', target)));
    plot_heatmap(freq, abs(results.(target).st - mlp_at_n.(target).st), parameter_names, ...
        sprintf('%s absolute MLP-teacher ST difference', target_labels{t}), ...
        fullfile(figures_dir, sprintf('%s_abs_st_difference_heatmap.png', target)));
end

metadata = struct();
metadata.schema_version = 1;
metadata.created_local = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
metadata.timezone = 'Europe/Berlin';
metadata.method = 'JCAL teacher Jansen Sobol validation';
metadata.dataset_run = config.dataset_run;
metadata.dataset_file = config.dataset_file;
metadata.shared_split_hash = char(split_data.split_hash);
metadata.mlp_sobol_file = config.mlp_sobol_file;
metadata.base_sample_size = n;
metadata.parameter_names = parameter_names;
metadata.lower_bounds = lower_bounds;
metadata.upper_bounds = upper_bounds;
metadata.fixed_inputs = struct('phi', phi, 'h_m', h);
metadata.input_distribution = 'independent linear-uniform in raw physical units';
metadata.sobol_skip = config.sobol_skip;
metadata.scramble_method = config.scramble_method;
metadata.sampling_seed = config.sampling_seed;
metadata.bootstrap_replicates = config.bootstrap_replicates;
metadata.bootstrap_seed = config.bootstrap_seed;
metadata.teacher_curve_evaluations = n * (d + 2);
metadata.output_postprocessing = 'none';
write_json(fullfile(config.output_dir, 'analysis_metadata.json'), metadata);

save(fullfile(config.output_dir, 'teacher_sobol_results.mat'), ...
    'results', 'mlp_at_n', 'freq', 'parameter_names', 'lower_bounds', ...
    'upper_bounds', 'phi', 'h', 'metadata', '-v7.3');
write_report(fullfile(config.output_dir, 'analysis_report.txt'), ...
    metadata, agreement_table, global_table);

fprintf('Teacher Sobol validation complete. Artifacts saved to: %s\n', config.output_dir);
end

function validate_contract(config, dataset, split_data, mlp_metadata, mlp_saved)
if ~strcmp(char(string(dataset.dataset_info.dataset_run)), config.dataset_run) || ...
        ~strcmp(char(string(split_data.dataset_run)), config.dataset_run)
    error('Teacher Sobol dataset/split run mismatch.');
end
if ~strcmp(char(string(dataset.dataset_info.complex_source)), 'Reflect')
    error('Teacher Sobol dataset target is not Reflect.');
end
if ~strcmp(char(string(mlp_metadata.dataset_run)), config.dataset_run) || ...
        ~strcmp(char(string(mlp_metadata.shared_split_hash)), char(string(split_data.split_hash)))
    error('MLP Sobol metadata is not aligned with the teacher validation run.');
end
if max(abs(double(mlp_saved.freq(:)) - double(dataset.freq_grid(:)))) > 1e-10
    error('MLP and teacher Sobol frequency grids do not match.');
end
end

function [air, h] = resolve_air_properties(dataset_info, phi)
porosity_cases = cellstr(string(dataset_info.selected_porosityfolders));
porosity_folder = porosity_cases{1};
if abs(str2double(porosity_folder) / 100 - phi) > 1e-12
    error('Teacher porosity does not match MLP Sobol metadata.');
end
[thickness, ~, pressure, ~, density, ~, ~, eta, gamma, c, ~, pr] = ...
    getFluidProperties(char(string(dataset_info.fiberfolder)), porosity_folder);
h = thickness * 1e-3;
air = struct('density_humid_air', density, 'speed_of_sound', c, ...
    'impedance', density * c, 'eta', eta, 'gamma', gamma, ...
    'Pr', pr, 'pressure', pressure);
end

function [y_re, y_im] = evaluate_teacher(material, phi, h, freq, air)
n = size(material, 1);
y_re = zeros(n, numel(freq));
y_im = zeros(n, numel(freq));
for i = 1:n
    reflect = jcal_reflection(h, phi, material(i, 1), material(i, 2), ...
        material(i, 3), material(i, 4), material(i, 5), freq, air);
    y_re(i, :) = real(reflect(:)).';
    y_im(i, :) = imag(reflect(:)).';
end
end

function selected = select_mlp_convergence(results, n, targets)
selected = struct();
for t = 1:numel(targets)
    runs = results.(targets{t}).convergence;
    match = find([runs.n] == n, 1);
    if isempty(match)
        error('MLP Sobol results do not contain N=%d.', n);
    end
    selected.(targets{t}) = runs(match);
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
        y_a(rows, :), y_b(rows, :), squared_first(rows, :, :), squared_total(rows, :, :));
end
s1_low = prctile(s1_boot, 2.5, 3);
s1_high = prctile(s1_boot, 97.5, 3);
st_low = prctile(st_boot, 2.5, 3);
st_high = prctile(st_boot, 97.5, 3);
end

function table_out = build_frequency_table(results, mlp, targets, labels, freq, parameters, n)
rows = {};
for t = 1:numel(targets)
    target = targets{t};
    for f_idx = 1:numel(freq)
        for j = 1:numel(parameters)
            rows(end + 1, :) = {target, labels{t}, freq(f_idx), parameters{j}, n, ... %#ok<AGROW>
                results.(target).s1(f_idx, j), results.(target).st(f_idx, j), ...
                results.(target).s1_ci_low(f_idx, j), results.(target).s1_ci_high(f_idx, j), ...
                results.(target).st_ci_low(f_idx, j), results.(target).st_ci_high(f_idx, j), ...
                mlp.(target).s1(f_idx, j), mlp.(target).st(f_idx, j), ...
                mlp.(target).s1(f_idx, j) - results.(target).s1(f_idx, j), ...
                mlp.(target).st(f_idx, j) - results.(target).st(f_idx, j)};
        end
    end
end
table_out = cell2table(rows, 'VariableNames', { ...
    'target', 'target_name', 'frequency_hz', 'parameter', 'base_sample_size', ...
    'teacher_S1', 'teacher_ST', 'teacher_S1_ci_low', 'teacher_S1_ci_high', ...
    'teacher_ST_ci_low', 'teacher_ST_ci_high', 'mlp_S1', 'mlp_ST', ...
    'mlp_minus_teacher_S1', 'mlp_minus_teacher_ST'});
end

function table_out = build_agreement_table(results, mlp, targets, labels)
rows = {};
for t = 1:numel(targets)
    target = targets{t};
    teacher_s1 = results.(target).s1;
    teacher_st = results.(target).st;
    mlp_s1 = mlp.(target).s1;
    mlp_st = mlp.(target).st;
    [~, teacher_top] = max(teacher_st, [], 2);
    [~, mlp_top] = max(mlp_st, [], 2);
    st_corr = corrcoef(teacher_st(:), mlp_st(:));
    s1_corr = corrcoef(teacher_s1(:), mlp_s1(:));
    rows(end + 1, :) = {target, labels{t}, ... %#ok<AGROW>
        sqrt(mean((mlp_s1 - teacher_s1).^2, 'all')), ...
        sqrt(mean((mlp_st - teacher_st).^2, 'all')), ...
        mean(abs(mlp_s1 - teacher_s1), 'all'), ...
        mean(abs(mlp_st - teacher_st), 'all'), ...
        max(abs(mlp_s1 - teacher_s1), [], 'all'), ...
        max(abs(mlp_st - teacher_st), [], 'all'), ...
        s1_corr(1, 2), st_corr(1, 2), mean(teacher_top == mlp_top)};
end
table_out = cell2table(rows, 'VariableNames', { ...
    'target', 'target_name', 'S1_RMSE', 'ST_RMSE', 'S1_MAE', 'ST_MAE', ...
    'S1_max_abs_difference', 'ST_max_abs_difference', ...
    'S1_flattened_correlation', 'ST_flattened_correlation', ...
    'top_ST_parameter_agreement_fraction'});
end

function table_out = build_functional_comparison( ...
        results, mlp, targets, labels, parameters, mask, region)
rows = {};
for t = 1:numel(targets)
    target = targets{t};
    teacher_variance = results.(target).output_variance(mask);
    mlp_variance = mlp.(target).variance(mask);
    for j = 1:numel(parameters)
        teacher_s1 = weighted_index(results.(target).s1(mask, j), teacher_variance);
        teacher_st = weighted_index(results.(target).st(mask, j), teacher_variance);
        mlp_s1 = weighted_index(mlp.(target).s1(mask, j), mlp_variance);
        mlp_st = weighted_index(mlp.(target).st(mask, j), mlp_variance);
        rows(end + 1, :) = {target, labels{t}, region, parameters{j}, ... %#ok<AGROW>
            teacher_s1, teacher_st, mlp_s1, mlp_st, ...
            mlp_s1 - teacher_s1, mlp_st - teacher_st};
    end
end
table_out = cell2table(rows, 'VariableNames', { ...
    'target', 'target_name', 'region', 'parameter', ...
    'teacher_functional_S1', 'teacher_functional_ST', ...
    'mlp_functional_S1', 'mlp_functional_ST', ...
    'mlp_minus_teacher_functional_S1', 'mlp_minus_teacher_functional_ST'});
end

function value = weighted_index(pointwise, variance)
value = sum(pointwise(:) .* variance(:), 'omitnan') / sum(variance, 'omitnan');
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

function write_json(path, value)
fid = fopen(path, 'w');
if fid < 0
    error('Could not open JSON output: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(value, 'PrettyPrint', true));
end

function write_report(path, metadata, agreement, global_table)
fid = fopen(path, 'w');
if fid < 0
    error('Could not open teacher Sobol report: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'JCAL Teacher Sobol Validation\n');
fprintf(fid, 'Dataset run: %s\n', metadata.dataset_run);
fprintf(fid, 'Base sample size: %d\n', metadata.base_sample_size);
fprintf(fid, 'Teacher curve evaluations: %d\n\n', metadata.teacher_curve_evaluations);
for i = 1:height(agreement)
    fprintf(fid, '%s agreement\n', upper(string(agreement.target{i})));
    fprintf(fid, 'S1 RMSE: %.6f | ST RMSE: %.6f\n', agreement.S1_RMSE(i), agreement.ST_RMSE(i));
    fprintf(fid, 'S1 correlation: %.6f | ST correlation: %.6f\n', ...
        agreement.S1_flattened_correlation(i), agreement.ST_flattened_correlation(i));
    fprintf(fid, 'Top-ST parameter agreement: %.2f%%\n\n', ...
        100 * agreement.top_ST_parameter_agreement_fraction(i));
end
for target = ["re", "im"]
    fprintf(fid, '%s teacher global functional indices\n', upper(target));
    subset = global_table(string(global_table.target) == target, :);
    for i = 1:height(subset)
        fprintf(fid, '%-20s S1=% .6f ST=% .6f | MLP ST=% .6f\n', ...
            subset.parameter{i}, subset.teacher_functional_S1(i), ...
            subset.teacher_functional_ST(i), subset.mlp_functional_ST(i));
    end
    if target ~= "im"
        fprintf(fid, '\n');
    end
end
end

function tf = directory_contains_files(path)
listing = dir(fullfile(path, '**', '*'));
tf = any(~[listing.isdir]);
end
