%% Paired Symbolic Dataset Inspection

clearvars -except symbolic_inspection_config;
clc;

script_dir = fileparts(mfilename('fullpath'));
run_config = segmented_dataset_run_config();
dataset_run = run_config.dataset_run;
dataset_file = run_config.segmented_dataset_file;
inspection_name = sprintf('%s_%s', char(datetime('now', 'Format', 'yyyyMMdd')), dataset_run);
artifacts_dir = fullfile(script_dir, 'artifacts', inspection_name);
scatter_max_points = 5000;

if exist('symbolic_inspection_config', 'var')
    if isfield(symbolic_inspection_config, 'dataset_run')
        run_config = segmented_dataset_run_config(symbolic_inspection_config.dataset_run);
        dataset_run = run_config.dataset_run; dataset_file = run_config.segmented_dataset_file;
    end
    if isfield(symbolic_inspection_config, 'dataset_file'), dataset_file = symbolic_inspection_config.dataset_file; end
    if isfield(symbolic_inspection_config, 'inspection_name'), inspection_name = symbolic_inspection_config.inspection_name; end
    if isfield(symbolic_inspection_config, 'artifacts_dir'), artifacts_dir = symbolic_inspection_config.artifacts_dir; end
    if isfield(symbolic_inspection_config, 'scatter_max_points'), scatter_max_points = symbolic_inspection_config.scatter_max_points; end
end
figures_dir = fullfile(artifacts_dir, 'figures');

assert_segmented_dataset_run_paths(dataset_run, dataset_file);
if ~exist(dataset_file, 'file'), error('Symbolic dataset file not found: %s', dataset_file); end
if ~exist(figures_dir, 'dir'), mkdir(figures_dir); end

d = load(dataset_file);
required = {'X_symbolic', 'y_re_symbolic', 'y_im_symbolic', 'source_curve_index', ...
    'segment_index', 'segment_definitions', 'symbolic_dataset_info'};
for i = 1:numel(required)
    if ~isfield(d, required{i}), error('Symbolic dataset is missing %s.', required{i}); end
end
n = size(d.X_symbolic, 1);
if size(d.X_symbolic, 2) ~= 8 || numel(d.y_re_symbolic) ~= n || numel(d.y_im_symbolic) ~= n || ...
        numel(d.source_curve_index) ~= n || numel(d.segment_index) ~= n
    error('Paired symbolic arrays and row metadata are not aligned.');
end
if any(~isfinite(d.X_symbolic), 'all') || any(~isfinite(d.y_re_symbolic)) || any(~isfinite(d.y_im_symbolic))
    error('Symbolic dataset contains non-finite values.');
end
names = cellstr(string(d.symbolic_dataset_info.target_names));
if ~isequal(names(:), {'R_real'; 'R_imag'}) || ...
        ~strcmp(char(string(d.symbolic_dataset_info.complex_source)), 'Reflect') || ...
        ~strcmp(char(string(d.symbolic_dataset_info.dataset_run)), dataset_run)
    error('Symbolic target/run metadata does not match the paired Reflect contract.');
end

summary = struct();
summary.dataset_run = dataset_run;
summary.num_symbolic_samples = n;
summary.num_curve_samples = numel(unique(d.source_curve_index));
summary.num_features = size(d.X_symbolic, 2);
summary.re = target_summary(d.y_re_symbolic);
summary.im = target_summary(d.y_im_symbolic);
summary.segment_counts = accumarray(d.segment_index(:), 1, [numel(d.segment_definitions), 1]);
save(fullfile(artifacts_dir, 'symbolic_dataset_inspection_summary.mat'), 'summary', '-v7.3');
write_report(fullfile(artifacts_dir, 'symbolic_dataset_inspection_report.txt'), summary, d.segment_definitions);
plot_target(d.X_symbolic(:, end), d.y_re_symbolic, d.segment_index, d.segment_definitions, ...
    'R_real', scatter_max_points, fullfile(figures_dir, 're_vs_frequency.png'));
plot_target(d.X_symbolic(:, end), d.y_im_symbolic, d.segment_index, d.segment_definitions, ...
    'R_imag', scatter_max_points, fullfile(figures_dir, 'im_vs_frequency.png'));

fprintf('Paired symbolic dataset inspection complete: %s\n', dataset_file);

function s = target_summary(y)
    s = struct('min', min(y), 'max', max(y), 'mean', mean(y), 'std', std(y, 0, 1));
end

function write_report(path, summary, definitions)
    fid = fopen(path, 'w'); if fid == -1, error('Could not write %s', path); end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, 'Paired Symbolic Dataset Inspection\nRun: %s\nCurves: %d\nRows: %d\nFeatures: %d\n', ...
        summary.dataset_run, summary.num_curve_samples, summary.num_symbolic_samples, summary.num_features);
    fprintf(fid, 'R_real: min=%.8e max=%.8e mean=%.8e std=%.8e\n', summary.re.min, summary.re.max, summary.re.mean, summary.re.std);
    fprintf(fid, 'R_imag: min=%.8e max=%.8e mean=%.8e std=%.8e\n', summary.im.min, summary.im.max, summary.im.mean, summary.im.std);
    for i = 1:numel(definitions)
        fprintf(fid, '%s: %d rows\n', definitions(i).name, summary.segment_counts(i));
    end
end

function plot_target(freq, y, segment_index, definitions, target_name, max_points, path)
    n = numel(y); rng(909); ids = 1:n;
    if n > max_points, ids = randperm(n, max_points); end
    fig = figure('Visible', 'off', 'Color', 'w'); hold on; colors = lines(numel(definitions));
    for seg = 1:numel(definitions)
        selected = ids(segment_index(ids) == seg);
        scatter(freq(selected), y(selected), 10, 'MarkerFaceColor', colors(seg, :), ...
            'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.3, 'DisplayName', definitions(seg).name);
    end
    xlabel('Frequency (Hz)'); ylabel(target_name, 'Interpreter', 'none');
    title(sprintf('%s Symbolic Samples', target_name), 'Interpreter', 'none'); legend('Location', 'best'); grid on;
    if exist('exportgraphics', 'file') == 2, exportgraphics(fig, path, 'Resolution', 200); else, saveas(fig, path); end
    close(fig);
end
