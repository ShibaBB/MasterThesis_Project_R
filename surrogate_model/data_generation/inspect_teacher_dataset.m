%% Paired Reflection Teacher Dataset Inspection
% Validates the R_real/R_imag contract and writes component diagnostics.

clearvars -except surrogate_inspection_config;
clc;

script_dir = fileparts(mfilename('fullpath'));
surrogate_root = fileparts(script_dir);
project_root = fileparts(surrogate_root);
addpath(surrogate_root, project_root);
original_dir = pwd;
cleanup_dir = onCleanup(@() cd(original_dir)); %#ok<NASGU>
cd(project_root);

requested_dataset_run = '';
if exist('surrogate_inspection_config', 'var') && isfield(surrogate_inspection_config, 'dataset_run')
    requested_dataset_run = surrogate_inspection_config.dataset_run;
end
run_config = resolve_dataset_run_config(requested_dataset_run);
dataset_file = run_config.paths.teacher_dataset_file;
inspection_name = sprintf('%s_%s', char(datetime('now', 'Format', 'yyyyMMdd')), char(string(run_config.run_id)));
artifacts_dir = fullfile(script_dir, 'artifacts', inspection_name);
num_random_curve_plots = 6;
num_physics_checks = 5;

if exist('surrogate_inspection_config', 'var')
    if isfield(surrogate_inspection_config, 'dataset_file'), dataset_file = surrogate_inspection_config.dataset_file; end
    if isfield(surrogate_inspection_config, 'inspection_name'), inspection_name = surrogate_inspection_config.inspection_name; end
    if isfield(surrogate_inspection_config, 'artifacts_dir'), artifacts_dir = surrogate_inspection_config.artifacts_dir; end
    if isfield(surrogate_inspection_config, 'num_random_curve_plots'), num_random_curve_plots = surrogate_inspection_config.num_random_curve_plots; end
    if isfield(surrogate_inspection_config, 'num_physics_checks'), num_physics_checks = surrogate_inspection_config.num_physics_checks; end
end
figures_dir = fullfile(artifacts_dir, 'figures');

if ~exist(dataset_file, 'file'), error('Dataset file not found: %s', dataset_file); end
if ~exist(figures_dir, 'dir'), mkdir(figures_dir); end

dataset = load(dataset_file);
required_variables = {'X', 'Y_re', 'Y_im', 'freq_grid', 'dataset_info', 'sample_metadata'};
for i = 1:numel(required_variables)
    if ~isfield(dataset, required_variables{i})
        error('Dataset file is missing required paired-target variable: %s', required_variables{i});
    end
end

X = dataset.X;
Y_re = dataset.Y_re;
Y_im = dataset.Y_im;
freq_grid = reshape(dataset.freq_grid, 1, []);
dataset_info = dataset.dataset_info;
sample_metadata = dataset.sample_metadata;

if size(X, 2) ~= 7, error('Teacher X must contain exactly seven features.'); end
if size(X, 1) ~= size(Y_re, 1) || ~isequal(size(Y_re), size(Y_im))
    error('X, Y_re, and Y_im must have aligned curve dimensions.');
end
if size(Y_re, 2) ~= numel(freq_grid)
    error('Paired target column count must match the frequency grid.');
end
if any(~isfinite(X), 'all') || any(~isfinite(Y_re), 'all') || any(~isfinite(Y_im), 'all')
    error('Teacher dataset contains non-finite values.');
end
validate_target_metadata(dataset_info, run_config);

max_physics_error = validate_against_reflect( ...
    X, Y_re, Y_im, freq_grid, sample_metadata, num_physics_checks);

feature_names = cellstr(string(dataset_info.parameter_names));
feature_summary = array2table([min(X, [], 1).', max(X, [], 1).', mean(X, 1).', std(X, 0, 1).'], ...
    'VariableNames', {'Min', 'Max', 'Mean', 'Std'}, 'RowNames', feature_names);
component_summary = struct();
component_summary.re = summarize_target(Y_re);
component_summary.im = summarize_target(Y_im);

inspection_summary = struct();
inspection_summary.dataset_run = char(string(dataset_info.dataset_run));
inspection_summary.target_names = {'R_real', 'R_imag'};
inspection_summary.num_samples = size(X, 1);
inspection_summary.num_features = size(X, 2);
inspection_summary.num_frequency_points = numel(freq_grid);
inspection_summary.feature_summary = feature_summary;
inspection_summary.component_summary = component_summary;
inspection_summary.max_checked_complex_error = max_physics_error;

save(fullfile(artifacts_dir, 'dataset_inspection_summary.mat'), 'inspection_summary', '-v7.3');
write_report(fullfile(artifacts_dir, 'dataset_inspection_report.txt'), inspection_summary, dataset_info);
plot_feature_histograms(X, feature_names, figures_dir);
plot_target_band(freq_grid, Y_re, 'R real', 'R_real', fullfile(figures_dir, 're_band.png'));
plot_target_band(freq_grid, Y_im, 'R imag', 'R_imag', fullfile(figures_dir, 'im_band.png'));
plot_random_curves(freq_grid, Y_re, 'R real', 'R_real', num_random_curve_plots, fullfile(figures_dir, 're_random_curves.png'));
plot_random_curves(freq_grid, Y_im, 'R imag', 'R_imag', num_random_curve_plots, fullfile(figures_dir, 'im_random_curves.png'));

fprintf('Paired teacher inspection complete: %s\n', dataset_file);
fprintf('Curves=%d | frequencies=%d | max checked complex error=%.3e\n', ...
    size(X, 1), numel(freq_grid), max_physics_error);

function validate_target_metadata(info, run_config)
    required = {'dataset_run', 'target_names', 'complex_source', 'target_definition'};
    for i = 1:numel(required)
        if ~isfield(info, required{i}), error('dataset_info is missing %s.', required{i}); end
    end
    names = cellstr(string(info.target_names));
    if ~isequal(names(:), {'R_real'; 'R_imag'}) || ~strcmp(char(string(info.complex_source)), 'Reflect')
        error('Teacher metadata does not describe paired Reflect components.');
    end
    if ~strcmp(char(string(info.dataset_run)), char(string(run_config.run_id)))
        error('Teacher dataset run does not match the selected run configuration.');
    end
    if ~strcmp(char(string(info.target_definition.R_real)), 'real(Reflect)') || ...
            ~strcmp(char(string(info.target_definition.R_imag)), 'imag(Reflect)')
        error('Teacher target definitions do not match real/imag(Reflect).');
    end
end

function max_error = validate_against_reflect(X, Y_re, Y_im, freq_grid, metadata, count)
    checked = min(count, size(X, 1));
    max_error = 0;
    for idx = 1:checked
        fiber = char(string(metadata(idx).fiberfolder));
        porosity = char(string(metadata(idx).porosityfolder));
        [~, ~, pressure, ~, density, ~, ~, eta, gamma, c, ~, Pr] = getFluidProperties(fiber, porosity);
        air = struct('density_humid_air', density, 'speed_of_sound', c, ...
            'impedance', density * c, 'eta', eta, 'gamma', gamma, 'Pr', Pr, 'pressure', pressure);
        [Reflect, ~, ~, ~, ~, ~, ~] = jcal_reflection( ...
            X(idx, 2), X(idx, 1), X(idx, 3), X(idx, 4), X(idx, 5), X(idx, 6), X(idx, 7), freq_grid, air);
        stored = Y_re(idx, :) + 1i * Y_im(idx, :);
        max_error = max(max_error, max(abs(stored - reshape(Reflect, 1, []))));
    end
    if max_error > 1e-12
        error('Stored targets differ from the checked complex Reflect values (max error %.3e).', max_error);
    end
end

function summary = summarize_target(Y)
    summary = struct('min', min(Y, [], 'all'), 'max', max(Y, [], 'all'), ...
        'mean', mean(Y, 'all'), 'std', std(Y, 0, 'all'), ...
        'curve_min', min(Y, [], 2), 'curve_max', max(Y, [], 2));
end

function write_report(path, summary, info)
    fid = fopen(path, 'w');
    if fid == -1, error('Could not open inspection report: %s', path); end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, 'Paired Reflection Teacher Dataset Inspection\n');
    fprintf(fid, 'Dataset run: %s\nMaterial: %s\nCurves: %d\nFrequency points: %d\n', ...
        summary.dataset_run, info.fiberfolder, summary.num_samples, summary.num_frequency_points);
    fprintf(fid, 'Complex source: Reflect\nR_real: real(Reflect)\nR_imag: imag(Reflect)\n');
    fprintf(fid, 'Maximum checked complex error: %.12e\n\n', summary.max_checked_complex_error);
    for target = {'re', 'im'}
        s = summary.component_summary.(target{1});
        fprintf(fid, '%s: min=%.8e max=%.8e mean=%.8e std=%.8e\n', upper(target{1}), s.min, s.max, s.mean, s.std);
    end
end

function plot_feature_histograms(X, names, figures_dir)
    fig = figure('Visible', 'off', 'Color', 'w');
    tiledlayout(size(X, 2), 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for i = 1:size(X, 2)
        nexttile; histogram(X(:, i), 30); xlabel(names{i}, 'Interpreter', 'none'); ylabel('Count'); grid on;
    end
    save_figure(fig, fullfile(figures_dir, 'feature_histograms.png')); close(fig);
end

function plot_target_band(freq, Y, label, symbol, path)
    avg = mean(Y, 1); p05 = prctile(Y, 5, 1); p95 = prctile(Y, 95, 1);
    fig = figure('Visible', 'off', 'Color', 'w');
    fill([freq, fliplr(freq)], [p05, fliplr(p95)], [0.85, 0.90, 1.00], 'EdgeColor', 'none'); hold on;
    plot(freq, avg, 'b-', 'LineWidth', 1.8); xlabel('Frequency (Hz)'); ylabel(symbol, 'Interpreter', 'none');
    title(sprintf('%s Mean and 5-95%% Band', label)); grid on; save_figure(fig, path); close(fig);
end

function plot_random_curves(freq, Y, label, symbol, count, path)
    rng(2024); ids = randperm(size(Y, 1), min(count, size(Y, 1)));
    fig = figure('Visible', 'off', 'Color', 'w'); plot(freq, Y(ids, :).', 'LineWidth', 1.1);
    xlabel('Frequency (Hz)'); ylabel(symbol, 'Interpreter', 'none'); title(sprintf('Random %s Curves', label)); grid on;
    save_figure(fig, path); close(fig);
end

function save_figure(fig, path)
    if exist('exportgraphics', 'file') == 2, exportgraphics(fig, path, 'Resolution', 200); else, saveas(fig, path); end
end
