%% Merge all inferred parameters
fiber_types = {'Acrylic', 'Silk', 'Wool'};
phi_values = 0.92:0.01:0.99;

% Step 1: Normalize all param files by replacing spaces with underscores
for f = 1:length(fiber_types)
    fiberfolder = fiber_types{f};
    for idx = 1:length(phi_values)
        porosity_str = num2str(round(phi_values(idx) * 100));
        filename = sprintf('%s_%s_params.txt', fiberfolder, porosity_str);

        if ~isfile(filename)
            continue;
        end

        content = fileread(filename);

        % Normalize parameter names
        content = regexprep(content, 'Airflow Resistivity', 'Airflow_Resistivity');
        content = regexprep(content, 'Viscous Characteristic Length', 'Viscous_Characteristic_Length');
        content = regexprep(content, 'Thermal Characteristic Length', 'Thermal_Characteristic_Length');
        content = regexprep(content, 'Static Thermal Permeability', 'Static_Thermal_Permeability');
        content = regexprep(content, 'Sigma Noise', 'Sigma_Noise');

        fid = fopen(filename, 'w');
        fwrite(fid, content);
        fclose(fid);
    end
end

% Step 2: Merge inferred parameters
param_names = {'Airflow_Resistivity', 'Tortuosity', ...
               'Viscous_Characteristic_Length', 'Thermal_Characteristic_Length', ...
               'Static_Thermal_Permeability', 'Sigma_Noise'};

num_params = length(param_names);
all_params = cell(length(fiber_types) * length(phi_values), 2 + 2 * num_params);
row_counter = 1;

for f = 1:length(fiber_types)
    fiberfolder = fiber_types{f};
    for idx = 1:length(phi_values)
        porosity_str = num2str(round(phi_values(idx) * 100));
        filename = sprintf('%s_%s_params.txt', fiberfolder, porosity_str);

        if ~isfile(filename)
            warning('File not found: %s (Skipping)', filename);
            continue;
        end

        fid = fopen(filename, 'r');
        param_values = zeros(1, num_params);
        param_sds = zeros(1, num_params);
        found_values = false;

        while ~feof(fid)
            line = strtrim(fgetl(fid));
            if contains(line, 'Inferred Parameters with Standard Deviation')
                found_values = true;
                fgetl(fid); % Skip the header line (e.g., "Parameter Name Value SD")
                continue;
            end
            if contains(line, 'Parameter Constraints')
                break;
            end
            if found_values
                tokens = regexp(line, '^(.+?)\s+([-+\deE\.]+)\s+([-+\deE\.]+)$', 'tokens');
                if ~isempty(tokens)
                    token = tokens{1};
                    param_name = strtrim(token{1});
                    value = str2double(token{2});
                    sd = str2double(token{3});
                    match_idx = find(strcmpi(param_names, param_name));
                    if ~isempty(match_idx)
                        param_values(match_idx) = value;
                        param_sds(match_idx) = sd;

                    else
                        warning('Unknown parameter name: "%s"', param_name);
                    end
                end
            end
        end

        fclose(fid);

        all_params(row_counter, :) = [fiberfolder, {phi_values(idx)}, ...
                                      num2cell(reshape([param_values; param_sds], 1, []))];
        row_counter = row_counter + 1;
    end
end

all_params = all_params(1:row_counter-1, :);

column_names = [{'Fiber Type', 'phi'}, ...
                reshape([param_names; strcat('SD_', param_names)], 1, [])];
params_table = cell2table(all_params, 'VariableNames', column_names);
disp('Extracted Parameters with SDs:');
disp(params_table);

output_filename = 'Extracted_All_Params.txt';
writetable(params_table, output_filename, 'Delimiter', 'tab');
fprintf('Saved extracted parameters to %s\n', output_filename);

%% Crelation between porosity and other five parameters (with SD)

% Fiber diameter
d_acrylic1=10e-6; 
d_acrylic2=40e-6;
d_silk1=4e-6;
d_silk2=15e-6;
d_wool1=16e-6;
d_wool2=40e-6;

data_inferred = readtable('Extracted_All_Params.txt', 'VariableNamingRule', 'preserve');

phi_for_regression=data_inferred.phi(1:8);

% Combined regression for three types fibers
figure;
tiledlayout(5, 3, 'TileSpacing', 'compact', 'Padding', 'none');

% --- Acrylic fiber: fitnlm with confidence bounds for airflow resistivity ---
ax1 = nexttile;

% Data
Acrylic_sigma = data_inferred.Airflow_Resistivity(1:8);
Acrylic_sigma_SD = data_inferred.SD_Airflow_Resistivity(1:8);
x = phi_for_regression(:);
y = Acrylic_sigma;

% Updated model: modelD
model = @(a1,x) a1(1) * (1 - x(:,1)) ./ (-0.640 * log(1 - x(:,1)) + 0.263 - x(:,1));
beta0 = 1e5;

% Fit
tbl = table(x, y, 'VariableNames', {'Phi', 'Sigma'});
nlm = fitnlm(tbl, model, beta0);

% Predictions with confidence intervals
phi_fitting = linspace(0.92, 0.99, 300)';
[ypred, yci] = predict(nlm, phi_fitting);

% --- Plot ---
hold(ax1, 'on');

% Error bars
errorbar(ax1, x, y, Acrylic_sigma_SD, 'o', ...
    'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', ...
    'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');

% Confidence bounds (shaded)
fill(ax1, [phi_fitting; flipud(phi_fitting)], ...
           [yci(:,1); flipud(yci(:,2))], ...
           [0.4 0.4 0.4], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

% Fitted curve
plot(ax1, phi_fitting, ypred, 'r', 'LineWidth', 4);

% Theoretical bounds
plot(ax1, phi_fitting, 16 * eta * (1 - phi_fitting) ./ (d_acrylic2^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting)), '--', 'LineWidth', 4);
plot(ax1, phi_fitting, 16 * eta * (1 - phi_fitting) ./ (d_acrylic1^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting)), '-.', 'LineWidth', 4);

% Axis and labels
ylabel(ax1, '\bf\sigma \rm (Pa s/m^2)', 'FontSize', 30);
xlim(ax1, [0.91 1]); 
ylim(ax1, [1e+2 1e+6]); 
xticklabels(ax1, {});

% Equation and R²
a1 = nlm.Coefficients.Estimate;
fittedvals = model(a1, x);
R_squared1 = corr(fittedvals, y)^2;

equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf('\\mathbf{%.2f \\cdot \\frac{(1 - \\phi)}{(-0.640 \\ln(1 - \\phi) - \\phi + 0.263)}}', a1);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax1, 0.04, 0.24, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');

r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared1);
text(ax1, 0.18, 0.08, ['$' r2str '$'], 'Units', 'normalized', ...
    'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

title(ax1, 'Acrylic fiber', 'FontSize', 18);

% --- Silk fiber: fitnlm with confidence bounds (dark gray), updated modelD ---
ax2 = nexttile;
Silk_sigma = data_inferred.Airflow_Resistivity(9:16);
Silk_sigma_SD = data_inferred.SD_Airflow_Resistivity(9:16);

% Data
x2 = phi_for_regression(:);
y2 = Silk_sigma;
y2_err = Silk_sigma_SD;

% Fit using modelD
model2 = @(a2, x) a2(1) * (1 - x) ./ (-0.640 * log(1 - x) + 0.263 - x);
beta0_2 = 1e5;
tbl2 = table(x2, y2);
nlm2 = fitnlm(tbl2, model2, beta0_2);
a2 = nlm2.Coefficients.Estimate;

% Predict with confidence intervals
phi_fitting = linspace(0.92, 0.99, 300)';
[ypred2, yci2] = predict(nlm2, phi_fitting);

% Plot
hold(ax2, 'on');
errorbar(ax2, x2, y2, y2_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax2, [phi_fitting; flipud(phi_fitting)], [yci2(:,1); flipud(yci2(:,2))], [0.5 0.5 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax2, phi_fitting, ypred2, 'r', 'LineWidth', 4);

% Upper and lower empirical bounds
plot(ax2, phi_fitting, 16 * eta * (1 - phi_fitting) ./ (d_silk2^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting)), '--', 'LineWidth', 4);
plot(ax2, phi_fitting, 16 * eta * (1 - phi_fitting) ./ (d_silk1^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting)), '-.', 'LineWidth', 4);

xlim(ax2, [0.91 1]);
ylim(ax2, [1e+3 5e+6]); 
xticklabels(ax2, {});

yfit2 = model2(a2, x2);
R_squared2 = corr(yfit2, y2)^2;

% Equation and R^2 in original style
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf('\\mathbf{%.2f \\cdot \\frac{(1 - \\phi)}{-0.640 \\ln(1 - \\phi) - \\phi + 0.263}}', a2);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax2, 0.02, 0.24, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared2);
text(ax2, 0.16, 0.08, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

title(ax2, 'Silk fiber', 'FontSize', 18);

% --- Wool fiber: fitnlm with confidence bounds (dark gray), updated modelD ---
ax3 = nexttile;
Wool_sigma = data_inferred.Airflow_Resistivity(17:24);
Wool_sigma_SD = data_inferred.SD_Airflow_Resistivity(17:24);

% Data
x3 = phi_for_regression(:);
y3 = Wool_sigma;
y3_err = Wool_sigma_SD;

% Fit using modelD
model3 = @(a3, x) a3(1) * (1 - x) ./ (-0.640 * log(1 - x) + 0.263 - x);
beta0_3 = 1e5;
tbl3 = table(x3, y3);
nlm3 = fitnlm(tbl3, model3, beta0_3);
a3 = nlm3.Coefficients.Estimate;

% Predict with confidence intervals
phi_fitting = linspace(0.92, 0.99, 300)';
[ypred3, yci3] = predict(nlm3, phi_fitting);

% Plot
hold(ax3, 'on');
errorbar(ax3, x3, y3, y3_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax3, [phi_fitting; flipud(phi_fitting)], [yci3(:,1); flipud(yci3(:,2))], [0.3 0.3 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax3, phi_fitting, ypred3, 'r', 'LineWidth', 4);

% Upper and lower empirical bounds
plot(ax3, phi_fitting, 16 * eta * (1 - phi_fitting) ./ (d_wool2^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting)), '--', 'LineWidth', 4);
plot(ax3, phi_fitting, 16 * eta * (1 - phi_fitting) ./ (d_wool1^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting)), '-.', 'LineWidth', 4);

xlim(ax3, [0.91 1]);
ylim(ax3, [1e2 1e6]);
xticklabels(ax3, {});

yfit3 = model3(a3, x3);
R_squared3 = corr(yfit3, y3)^2;

% Equation and R^2 in original style
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf('\\mathbf{%.2f \\cdot \\frac{(1 - \\phi)}{-0.640 \\ln(1 - \\phi) + 0.263 - \\phi}}', a3);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax3, 0.02, 0.24, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared3);
text(ax3, 0.16, 0.08, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

title(ax3,'Wool fiber','FontSize', 18);

% --- Acrylic fiber: fitnlm with confidence bounds for tortuosity ---
ax4 = nexttile;
Acrylic_tortuosity = data_inferred.Tortuosity(1:8);
Acrylic_tortuosity_SD = data_inferred.SD_Tortuosity(1:8);

% Data
x4 = phi_for_regression(:);
y4 = Acrylic_tortuosity;
y4_err = Acrylic_tortuosity_SD;

% Model: y = 1 - b1 * ln(phi)
model4 = @(b, x) 1 - b(1) * log(x);
beta0_4 = 1;  % initial guess
tbl4 = table(x4, y4);
nlm4 = fitnlm(tbl4, model4, beta0_4);
b1 = nlm4.Coefficients.Estimate;  % store b1

% Predict with confidence intervals
[ypred4, yci4] = predict(nlm4, phi_fitting);

% Plot
hold(ax4, 'on');
errorbar(ax4, x4, y4, y4_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax4, [phi_fitting; flipud(phi_fitting)], [yci4(:,1); flipud(yci4(:,2))], [0.3 0.3 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax4, phi_fitting, ypred4, 'r', 'LineWidth', 4);

% Upper and lower bounds
plot(ax4, phi_fitting, 1 + phi_fitting - phi_fitting, '--', 'LineWidth', 4);  % lower bound
plot(ax4, phi_fitting, 3 - phi_fitting, '-.', 'LineWidth', 4);  % upper bound

% Axis and labels
ylabel(ax4, '\bf\alpha_{\infty} \rm', 'FontSize', 30);
xlim(ax4, [0.91 1]);
ylim(ax4, [1 2.6]);
xticklabels(ax4, {});

% Equation and R^2 in original style
yfit4 = model4(b1, x4);
R_squared4 = corr(yfit4, y4)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf('\\mathbf{1 - %.2f \\cdot \\ln(\\phi)}', b1);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax4, 0.5, 0.88, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared4);
text(ax4, 0.64, 0.75, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% --- Silk fiber: fitnlm with confidence bounds for tortuosity ---
ax5 = nexttile;
Silk_tortuosity = data_inferred.Tortuosity(9:16);
Silk_tortuosity_SD = data_inferred.SD_Tortuosity(9:16);

% Data
x5 = phi_for_regression(:);
y5 = Silk_tortuosity;
y5_err = Silk_tortuosity_SD;

% Model: y = 1 - b * ln(phi)
model5 = @(b, x) 1 - b(1) * log(x);
beta0_5 = 1;
tbl5 = table(x5, y5);
nlm5 = fitnlm(tbl5, model5, beta0_5);
b2 = nlm5.Coefficients.Estimate;

% Predict with confidence intervals
[ypred5, yci5] = predict(nlm5, phi_fitting);

% Plot
hold(ax5, 'on');
errorbar(ax5, x5, y5, y5_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax5, [phi_fitting; flipud(phi_fitting)], [yci5(:,1); flipud(yci5(:,2))], [0.3 0.3 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax5, phi_fitting, ypred5, 'r', 'LineWidth', 4);

% Upper and lower bounds
plot(ax5, phi_fitting, 1 + phi_fitting - phi_fitting, '--', 'LineWidth', 4);
plot(ax5, phi_fitting, 3 - phi_fitting, '-.', 'LineWidth', 4);

% Axis and labels
xlim(ax5, [0.91 1]);
xticklabels(ax5, {});

% Equation and R^2 in original style
yfit5 = model5(b2, x5);
R_squared5 = corr(yfit5, y5)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf('\\mathbf{1 - %.2f \\cdot \\ln(\\phi)}', b2);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax5, 0.5, 0.88, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared5);
text(ax5, 0.64, 0.75, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% --- Wool fiber: fitnlm with confidence bounds for tortuosity ---
ax6 = nexttile;
Wool_tortuosity = data_inferred.Tortuosity(17:24);
Wool_tortuosity_SD = data_inferred.SD_Tortuosity(17:24);

% Data
x6 = phi_for_regression(:);
y6 = Wool_tortuosity;
y6_err = Wool_tortuosity_SD;

% Model: y = 1 - b * ln(phi)
model6 = @(b, x) 1 - b(1) * log(x);
beta0_6 = 1;
tbl6 = table(x6, y6);
nlm6 = fitnlm(tbl6, model6, beta0_6);
b3 = nlm6.Coefficients.Estimate;

% Predict with confidence intervals
[ypred6, yci6] = predict(nlm6, phi_fitting);

% Plot
hold(ax6, 'on');
errorbar(ax6, x6, y6, y6_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax6, [phi_fitting; flipud(phi_fitting)], [yci6(:,1); flipud(yci6(:,2))], [0.3 0.3 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax6, phi_fitting, ypred6, 'r', 'LineWidth', 4);

% Upper and lower bounds
plot(ax6, phi_fitting, 1 + phi_fitting - phi_fitting, '--', 'LineWidth', 4);
plot(ax6, phi_fitting, 3 - phi_fitting, '-.', 'LineWidth', 4);

% Axis and labels
xlim(ax6, [0.91 1]);
ylim(ax6, [1 2.6]);
xticklabels(ax6, {});

% Equation and R^2 in original style
yfit6 = model6(b3, x6);
R_squared6 = corr(yfit6, y6)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf('\\mathbf{1 - %.2f \\cdot \\ln(\\phi)}', b3);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax6, 0.5, 0.88, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared6);
text(ax6, 0.65, 0.75, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% --- Acrylic fiber: fitnlm with confidence bounds for viscous characteristic length ---
ax7 = nexttile;
Acrylic_lambda = data_inferred.Viscous_Characteristic_Length(1:8);
Acrylic_lambda_SD = data_inferred.SD_Viscous_Characteristic_Length(1:8);

% Data
x7 = phi_for_regression(:);
y7 = Acrylic_lambda * 1e6;  % Convert to microns
y7_err = Acrylic_lambda_SD * 1e6;

% Updated Model: y = c1 * sqrt(((1 - b1*log(phi)) * (-0.64*log(1-phi)-phi+0.263)) / (phi*(1 - phi)))
model7 = @(c, x) c(1) * sqrt(((1 - b1 * log(x)) .* (-0.64 * log(1 - x) - x + 0.263)) ./ ((1 - x) .* x));
beta0_7 = 1;
tbl7 = table(x7, y7);
nlm7 = fitnlm(tbl7, model7, beta0_7);
c1 = nlm7.Coefficients.Estimate;

% Predictions with confidence intervals
[ypred7, yci7] = predict(nlm7, phi_fitting);

% Plot
hold(ax7, 'on');
errorbar(ax7, x7, y7, y7_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', ...
    'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax7, [phi_fitting; flipud(phi_fitting)], ...
     [yci7(:,1); flipud(yci7(:,2))], [0.3 0.3 0.3], ...
     'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax7, phi_fitting, ypred7, 'r', 'LineWidth', 4);

% Theoretical bounds
plot(ax7, phi_fitting, (8 * eta ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_acrylic1^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '--', 'LineWidth', 4);
plot(ax7, phi_fitting, (8 * eta * (3 - phi_fitting) ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_acrylic2^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '-.', 'LineWidth', 4);

% Axis and labels
ylabel(ax7, '\bf\Lambda \rm (\mum)', 'FontSize', 30);
xlim(ax7, [0.91 1]);
ylim(ax7, [20 2e+3]);
xticklabels(ax7, {});

% Equation and R²
yfit7 = model7(c1, x7);
R_squared7 = corr(yfit7, y7)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf('\\mathbf{%.2f \\cdot \\sqrt{\\frac{(1 - %.2f \\cdot \\ln(\\phi)) ( -0.64 \\cdot \\ln(1 - \\phi) - \\phi + 0.263 )}{\\phi \\cdot (1 - \\phi)}}}', c1, b1);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax7, 0.01, 0.8, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared7);
text(ax7, 0.12, 0.6, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% --- Silk fiber: fitnlm with confidence bounds for viscous characteristic length ---
ax8 = nexttile;
Silk_lambda = data_inferred.Viscous_Characteristic_Length(9:16);
Silk_lambda_SD = data_inferred.SD_Viscous_Characteristic_Length(9:16);

% Data
x8 = phi_for_regression(:);
y8 = Silk_lambda * 1e6;
y8_err = Silk_lambda_SD * 1e6;

% Updated Model: y = c1 * sqrt(((1 - b2*log(phi)) * (-0.64*log(1-phi)-phi+0.263)) / (phi*(1 - phi)))
model8 = @(c, x) c(1) * sqrt(((1 - b2 * log(x)) .* (-0.64 * log(1 - x) - x + 0.263)) ./ (x .* (1 - x)));
beta0_8 = 1;
tbl8 = table(x8, y8);
nlm8 = fitnlm(tbl8, model8, beta0_8);
c2 = nlm8.Coefficients.Estimate;

% Predictions with confidence intervals
[ypred8, yci8] = predict(nlm8, phi_fitting);

% Plot
hold(ax8, 'on');
errorbar(ax8, x8, y8, y8_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', ...
    'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax8, [phi_fitting; flipud(phi_fitting)], ...
     [yci8(:,1); flipud(yci8(:,2))], [0.3 0.3 0.3], ...
     'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax8, phi_fitting, ypred8, 'r', 'LineWidth', 4);

% Theoretical bounds
plot(ax8, phi_fitting, (8 * eta ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_silk1^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '--', 'LineWidth', 4);
plot(ax8, phi_fitting, (8 * eta * (3 - phi_fitting) ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_silk2^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '-.', 'LineWidth', 4);

% Axis
xlim(ax8, [0.91 1]);
ylim(ax8, [1 2e+3]);
xticklabels(ax8, {});

% Equation and R²
yfit8 = model8(c2, x8);
R_squared8 = corr(yfit8, y8)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf(['\\mathbf{%.2f \\cdot \\sqrt{\\frac{(1 - %.2f \\cdot \\ln(\\phi))' ...
    '\\cdot (-0.64 \\cdot \\ln(1 - \\phi) - \\phi + 0.263)}{\\phi (1 - \\phi)}}}'], c2, b2);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax8, 0.01, 0.8, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared8);
text(ax8, 0.12, 0.6, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% --- Wool fiber: fitnlm with confidence bounds for viscous characteristic length ---
ax9 = nexttile;
Wool_lambda = data_inferred.Viscous_Characteristic_Length(17:24);
Wool_lambda_SD = data_inferred.SD_Viscous_Characteristic_Length(17:24);

% Data
x9 = phi_for_regression(:);
y9 = Wool_lambda * 1e6;
y9_err = Wool_lambda_SD * 1e6;

% Updated Model: y = c3 * sqrt(((1 - b3*log(phi)) * (-0.64*log(1-phi)-phi+0.263)) / (phi*(1 - phi)))
model9 = @(c, x) c(1) * sqrt(((1 - b3 * log(x)) .* (-0.64 * log(1 - x) - x + 0.263)) ./ (x .* (1 - x)));
beta0_9 = 1;
tbl9 = table(x9, y9);
nlm9 = fitnlm(tbl9, model9, beta0_9);
c3 = nlm9.Coefficients.Estimate;

% Predictions with confidence intervals
[ypred9, yci9] = predict(nlm9, phi_fitting);

% Plot
hold(ax9, 'on');
errorbar(ax9, x9, y9, y9_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', ...
    'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax9, [phi_fitting; flipud(phi_fitting)], [yci9(:,1); flipud(yci9(:,2))], ...
    [0.3 0.3 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax9, phi_fitting, ypred9, 'r', 'LineWidth', 4);

% Theoretical bounds
plot(ax9, phi_fitting, (8 * eta ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_wool1^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '--', 'LineWidth', 4);
plot(ax9, phi_fitting, (8 * eta * (3 - phi_fitting) ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_wool2^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '-.', 'LineWidth', 4);

% Axis
xlim(ax9, [0.91 1]);
xticklabels(ax9, {});

% Equation and R²
yfit9 = model9(c3, x9);
R_squared9 = corr(yfit9, y9)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf(['\\mathbf{%.2f \\cdot \\sqrt{\\frac{(1 - %.2f \\cdot \\ln(\\phi))' ...
    '\\cdot (-0.64 \\cdot \\ln(1 - \\phi) - \\phi + 0.263)}{\\phi (1 - \\phi)}}}'], c3, b3);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax9, 0.01, 0.8, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared9);
text(ax9, 0.12, 0.6, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% --- Acrylic fiber: fitnlm with confidence bounds for thermal characteristic length ---
ax10 = nexttile;
Acrylic_lambda_prime = data_inferred.Thermal_Characteristic_Length(1:8);
Acrylic_lambda_prime_SD = data_inferred.SD_Thermal_Characteristic_Length(1:8);

% Data
x10 = phi_for_regression(:);
y10 = Acrylic_lambda_prime * 1e6;
y10_err = Acrylic_lambda_prime_SD * 1e6;

% Updated Model: y = d1 * sqrt(((1 - b1*log(phi)) * (-0.64*log(1-phi)-phi+0.263)) / (phi*(1 - phi)))
model10 = @(d, x) d(1) * sqrt(((1 - b1 * log(x)) .* (-0.64 * log(1 - x) - x + 0.263)) ./ (x .* (1 - x)));
beta0_10 = 1;
tbl10 = table(x10, y10);
nlm10 = fitnlm(tbl10, model10, beta0_10);
d1 = nlm10.Coefficients.Estimate;

% Predictions with confidence intervals
[ypred10, yci10] = predict(nlm10, phi_fitting);

% Plot
hold(ax10, 'on');
errorbar(ax10, x10, y10, y10_err, 'o', ...
    'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', ...
    'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax10, [phi_fitting; flipud(phi_fitting)], ...
     [yci10(:,1); flipud(yci10(:,2))], [0.3 0.3 0.3], ...
     'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax10, phi_fitting, ypred10, 'r', 'LineWidth', 4);

% Theoretical bounds
plot(ax10, phi_fitting, 2 * (8 * eta ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_acrylic1^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '--', 'LineWidth', 4);
plot(ax10, phi_fitting, 2 * (8 * eta * (3 - phi_fitting) ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_acrylic2^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '-.', 'LineWidth', 4);

% Axis and labels
ylabel(ax10, '\bf\Lambda^{\prime} \rm (\mum)', 'FontSize', 30);
xlim(ax10, [0.91 1]);
ylim(ax10, [5e+1 5e+3]);
xticklabels(ax10, {});

% Equation and R^2
yfit10 = model10(d1, x10);
R_squared10 = corr(yfit10, y10)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf(['\\mathbf{%.2f \\cdot \\sqrt{\\frac{(1 - %.2f \\cdot \\ln(\\phi))' ...
    '\\cdot (-0.64 \\cdot \\ln(1 - \\phi) - \\phi + 0.263)}{\\phi (1 - \\phi)}}}'], d1, b1);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax10, 0.01, 0.8, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared10);
text(ax10, 0.12, 0.6, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% --- Silk fiber: fitnlm with confidence bounds for thermal characteristic length ---
ax11 = nexttile;
Silk_lambda_prime = data_inferred.Thermal_Characteristic_Length(9:16);
Silk_lambda_prime_SD = data_inferred.SD_Thermal_Characteristic_Length(9:16);

% Data
x11 = phi_for_regression(:);
y11 = Silk_lambda_prime * 1e6;
y11_err = Silk_lambda_prime_SD * 1e6;

% Updated Model: y = d2 * sqrt(((1 - b2*log(phi)) * (-0.64*log(1-phi)-phi+0.263)) / (phi*(1 - phi)))
model11 = @(d, x) d(1) * sqrt(((1 - b2 * log(x)) .* (-0.64 * log(1 - x) - x + 0.263)) ./ (x .* (1 - x)));
beta0_11 = 1;
tbl11 = table(x11, y11);
nlm11 = fitnlm(tbl11, model11, beta0_11);
d2 = nlm11.Coefficients.Estimate;

% Predictions with confidence intervals
[ypred11, yci11] = predict(nlm11, phi_fitting);

% Plot
hold(ax11, 'on');
errorbar(ax11, x11, y11, y11_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', ...
    'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax11, [phi_fitting; flipud(phi_fitting)], [yci11(:,1); flipud(yci11(:,2))], ...
    [0.3 0.3 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax11, phi_fitting, ypred11, 'r', 'LineWidth', 4);

% Theoretical bounds
plot(ax11, phi_fitting, 2* (8 * eta ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_silk1^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '--', 'LineWidth', 4);
plot(ax11, phi_fitting, 2* (8 * eta * (3 - phi_fitting) ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_silk2^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '-.', 'LineWidth', 4);

% Axis and labels
xlim(ax11, [0.91 1]);
ylim(ax11, [20 2e+3]);
xticklabels(ax11, {});

% Equation and R^2
yfit11 = model11(d2, x11);
R_squared11 = corr(yfit11, y11)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf(['\\mathbf{%.2f \\cdot \\sqrt{\\frac{(1 - %.2f \\cdot \\ln(\\phi))' ...
    '\\cdot (-0.64 \\cdot \\ln(1 - \\phi) - \\phi + 0.263)}{\\phi (1 - \\phi)}}}'], d2, b2);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax11, 0.01, 0.8, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared11);
text(ax11, 0.12, 0.6, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% --- Wool fiber: fitnlm with confidence bounds for thermal characteristic length ---
ax12 = nexttile;
Wool_lambda_prime = data_inferred.Thermal_Characteristic_Length(17:24);
Wool_lambda_prime_SD = data_inferred.SD_Thermal_Characteristic_Length(17:24);

% Data
x12 = phi_for_regression(:);
y12 = Wool_lambda_prime * 1e6;
y12_err = Wool_lambda_prime_SD * 1e6;

% Updated Model: y = d3 * sqrt(((1 - b3*log(phi)) * (-0.64*log(1-phi)-phi+0.263)) / (phi*(1 - phi)))
model12 = @(d, x) d(1) * sqrt(((1 - b3 * log(x)) .* (-0.64 * log(1 - x) - x + 0.263)) ./ (x .* (1 - x)));
beta0_12 = 1;
tbl12 = table(x12, y12);
nlm12 = fitnlm(tbl12, model12, beta0_12);
d3 = nlm12.Coefficients.Estimate;

% Predictions with confidence intervals
[ypred12, yci12] = predict(nlm12, phi_fitting);

% Plot
hold(ax12, 'on');
errorbar(ax12, x12, y12, y12_err, 'o', ...
    'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', ...
    'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax12, [phi_fitting; flipud(phi_fitting)], ...
     [yci12(:,1); flipud(yci12(:,2))], [0.3 0.3 0.3], ...
     'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax12, phi_fitting, ypred12, 'r', 'LineWidth', 4);

% Theoretical bounds
plot(ax12, phi_fitting, 2 * (8 * eta ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_wool1^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '--', 'LineWidth', 4);
plot(ax12, phi_fitting, 2 * (8 * eta * (3 - phi_fitting) ./ ((phi_fitting) .* (16 * eta * (1 - phi_fitting) ./ (d_wool2^2 * (-0.640 * log(1 - phi_fitting) + 0.263 - phi_fitting))))) .^ 0.5 * 1e6, '-.', 'LineWidth', 4);

% Axis and labels
xlim(ax12, [0.91 1]);
ylim(ax12, [50 5e+3]);
xticklabels(ax12, {});

% Equation and R^2
yfit12 = model12(d3, x12);
R_squared12 = corr(yfit12, y12)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf(['\\mathbf{%.2f \\cdot \\sqrt{\\frac{(1 - %.2f \\cdot \\ln(\\phi))' ...
    '\\cdot (-0.64 \\cdot \\ln(1 - \\phi) - \\phi + 0.263)}{\\phi (1 - \\phi)}}}'], d3, b3);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax12, 0.01, 0.8, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared12);
text(ax12, 0.12, 0.6, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% --- Acrylic fiber: fitnlm with confidence bounds for thermal permeability ---
ax13 = nexttile;
Acrylic_k0_prime = data_inferred.Static_Thermal_Permeability(1:8);
Acrylic_k0_prime_SD = data_inferred.SD_Static_Thermal_Permeability(1:8);

% Data
x13 = phi_for_regression(:);
y13 = Acrylic_k0_prime * 1e10;
y13_err = Acrylic_k0_prime_SD * 1e10;

% Model and fit
model13 = @(e, x) e(1) * (x.^2 ./ (1 - x));
beta0_13 = 0.1;
tbl13 = table(x13, y13);
nlm13 = fitnlm(tbl13, model13, beta0_13);
e1 = nlm13.Coefficients.Estimate;

% Predictions with confidence intervals
[ypred13, yci13] = predict(nlm13, phi_fitting);

% Plot
hold(ax13, 'on');
errorbar(ax13, x13, y13, y13_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax13, [phi_fitting; flipud(phi_fitting)], [yci13(:,1); flipud(yci13(:,2))], [0.3 0.3 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax13, phi_fitting, ypred13, 'r', 'LineWidth', 4);

% Theoretical bounds
plot(ax13, phi_fitting, ones(size(phi_fitting)), '--', 'LineWidth', 4);         % Lower bound at 1
plot(ax13, phi_fitting, 100 * ones(size(phi_fitting)), '-.', 'LineWidth', 4);  % Upper bound at 100


% Axis and labels
xlabel(ax13, 'Porosity', 'FontSize', 30);
ylabel(ax13, '\bfk^{\prime}_{0} \rm (10^{-10} m^2)', 'FontSize', 30);
xlim(ax13, [0.91 1]);
ylim(ax13, [0.1 1e+4]);

% Equation and R^2
yfit13 = model13(e1, x13);
R_squared13 = corr(yfit13, y13)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf('\\mathbf{%.2f \\cdot \\frac{\\phi^2}{(1 - \\phi)}}', e1);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax13, 0.02, 0.88, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared13);
text(ax13, 0.14, 0.75, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% --- Silk fiber: fitnlm with confidence bounds for thermal permeability ---
ax14 = nexttile;
Silk_k0_prime = data_inferred.Static_Thermal_Permeability(9:16);
Silk_k0_prime_SD = data_inferred.SD_Static_Thermal_Permeability(9:16);

% Data
x14 = phi_for_regression(:);
y14 = Silk_k0_prime * 1e10;
y14_err = Silk_k0_prime_SD * 1e10;

% Model and fit: y = e * phi^2 / (1 - phi)
model14 = @(e, x) e(1) * (x.^2 ./ (1 - x));
beta0_14 = 0.1;
tbl14 = table(x14, y14);
nlm14 = fitnlm(tbl14, model14, beta0_14);
e2 = nlm14.Coefficients.Estimate;

% Predictions with confidence intervals
[ypred14, yci14] = predict(nlm14, phi_fitting);

% Plot
hold(ax14, 'on');
errorbar(ax14, x14, y14, y14_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', ...
    'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax14, [phi_fitting; flipud(phi_fitting)], [yci14(:,1); flipud(yci14(:,2))], ...
    [0.3 0.3 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax14, phi_fitting, ypred14, 'r', 'LineWidth', 4);

% Theoretical bounds
plot(ax14, phi_fitting, ones(size(phi_fitting)), '--', 'LineWidth', 4);         % Lower bound at 1
plot(ax14, phi_fitting, 100 * ones(size(phi_fitting)), '-.', 'LineWidth', 4);  % Upper bound at 100

% Axis and labels
xlabel(ax14, 'Porosity', 'FontSize', 30);
xlim(ax14, [0.91 1]);
ylim(ax14, [0.1 1e+4]);

% Equation and R^2
yfit14 = model14(e2, x14);
R_squared14 = corr(yfit14, y14)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf('\\mathbf{%.2f \\cdot \\frac{\\phi^2}{(1 - \\phi)}}', e2);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax14, 0.02, 0.88, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared14);
text(ax14, 0.14, 0.75, ['$' r2str '$'], 'Units', 'normalized', ...
    'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% --- Wool fiber: fitnlm with confidence bounds for thermal permeability ---
ax15 = nexttile;
Wool_k0_prime = data_inferred.Static_Thermal_Permeability(17:24);
Wool_k0_prime_SD = data_inferred.SD_Static_Thermal_Permeability(17:24);

% Data
x15 = phi_for_regression(:);
y15 = Wool_k0_prime * 1e10;
y15_err = Wool_k0_prime_SD * 1e10;

% Model and fit: y = e * phi^2 / (1 - phi)
model15 = @(e, x) e(1) * (x.^2 ./ (1 - x));
beta0_15 = 0.1;
tbl15 = table(x15, y15);
nlm15 = fitnlm(tbl15, model15, beta0_15);
e3 = nlm15.Coefficients.Estimate;

% Predictions with confidence intervals
[ypred15, yci15] = predict(nlm15, phi_fitting);

% Plot
hold(ax15, 'on');
errorbar(ax15, x15, y15, y15_err, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', ...
    'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, 'LineStyle', 'none');
fill(ax15, [phi_fitting; flipud(phi_fitting)], [yci15(:,1); flipud(yci15(:,2))], ...
    [0.3 0.3 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(ax15, phi_fitting, ypred15, 'r', 'LineWidth', 4);

% Theoretical bounds
plot(ax15, phi_fitting, ones(size(phi_fitting)), '--', 'LineWidth', 4);         % Lower bound at 1
plot(ax15, phi_fitting, 100 * ones(size(phi_fitting)), '-.', 'LineWidth', 4);  % Upper bound at 100

% Axis and labels
xlabel(ax15, 'Porosity', 'FontSize', 30);
xlim(ax15, [0.91 1]);
ylim(ax15, [0.1 1e+4]);

% Equation and R^2
yfit15 = model15(e3, x15);
R_squared15 = corr(yfit15, y15)^2;
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf('\\mathbf{%.2f \\cdot \\frac{\\phi^2}{(1 - \\phi)}}', e3);
equation = ['$' equation_part1 ' ' equation_part2 '$'];

text(ax15, 0.08, 0.88, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');
r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared15);
text(ax15, 0.2, 0.75, ['$' r2str '$'], 'Units', 'normalized', 'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

% sgtitle('Correlation between porosity and other non-acoustical parameters in JCAL model','FontSize', 20);
set(gcf, 'Position', get(0, 'Screensize'));  % Maximize figure window
hAx = findobj(gcf, 'type', 'axes');
for i = 1:numel(hAx)
    set(hAx(i), 'LineWidth', 1, 'FontSize', 30);
    grid(hAx(i), 'on');
    box(hAx(i), 'on');
    set(hAx(i), 'YScale', 'log'); % Set y-axis to log scale
end

hold off;

set(ax4, 'YScale', 'linear');
set(ax5, 'YScale', 'linear');
set(ax6, 'YScale', 'linear');

%% Legends for the combined regression figure
% --- Acrylic fiber: fitnlm with confidence bounds for airflow resistivity ---
figure;

% Main plot with actual data and models
% Error bars
h1 = errorbar(x, y, Acrylic_sigma_SD, 'o', ...
    'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', ...
    'MarkerSize', 4, 'LineWidth', 2, 'CapSize', 20, ...
    'LineStyle', 'none', 'DisplayName', 'Inferred values');

hold on;

% Confidence bounds (shaded)
phi_fitting = phi_fitting(:); % Ensure it's a column vector
h2 = fill([phi_fitting; flipud(phi_fitting)], ...
          [yci(:,1); flipud(yci(:,2))], ...
          [0.4 0.4 0.4], ...
          'EdgeColor', 'none', 'FaceAlpha', 0.5, ...
          'DisplayName', '95% Confidence interval');

% Fitted curve
h3 = plot(phi_fitting, ypred, 'r', 'LineWidth', 4, 'DisplayName', 'Fitted curve');

% Theoretical bounds
h4 = plot(phi_fitting, ...
    (180 * eta * (1 - phi_fitting).^2) ./ (d_acrylic2^2 * phi_fitting.^3), ...
    '--', 'LineWidth', 4, 'DisplayName', 'Lower bound');

h5 = plot(phi_fitting, ...
    (180 * eta * (1 - phi_fitting).^2) ./ (d_acrylic1^2 * phi_fitting.^3), ...
    '-.', 'LineWidth', 4, 'DisplayName', 'Upper bound');

% Axis and labels
ylabel('\bf\sigma \rm (Pa s/m^2)', 'FontSize', 30);
xlim([0.91 1]);
set(gca, 'YScale', 'log');
set(gca, 'FontSize', 26);

% Equation and R^2
equation_part1 = '\textnormal{\textbf{Fitted: }}\mathbf{y} =';
equation_part2 = sprintf('\\mathbf{%.2f (1 - \\phi)^2 / \\phi^3}', a1);
equation = ['$' equation_part1 ' ' equation_part2 '$'];
text(0.04, 0.24, equation, 'Units', 'normalized', 'FontSize', 26, 'Interpreter', 'latex');

r2str = sprintf('\\mathbf{R^2 = %.2f}', R_squared1);
text(0.18, 0.08, ['$' r2str '$'], 'Units', 'normalized', ...
    'Color', [0 0 0], 'FontSize', 26, 'Interpreter', 'latex');

title('Acrylic fiber', 'FontSize', 18);

% External legend
legend([h1 h3 h4 h5 h2], ...
    'Location', 'eastoutside', ...
    'Orientation', 'vertical', ...
    'Box', 'off', ...
    'FontSize', 24);


%% Save coefficients in the simplfied JCAL models
coefficients = struct();
% coefficients for airflow resistivity
coefficients.a1 = a1;
coefficients.a2 = a2;
coefficients.a3 = a3;
% coefficients for tortuosity
coefficients.b1 = b1;
coefficients.b2 = b2;
coefficients.b3 = b3;
% coefficients for viscous characteristic length
coefficients.c1 = c1;
coefficients.c2 = c2;
coefficients.c3 = c3;
% coefficients for thermal characteristic length
coefficients.d1 = d1;
coefficients.d2 = d2;
coefficients.d3 = d3;
% coefficients for thermal permeability
coefficients.e1 = e1;
coefficients.e2 = e2;
coefficients.e3 = e3;

%% Comparison for 95% Porosity
% Define Fiber Types and Porosity
fiberTypes = {'Acrylic', 'Silk', 'Wool'};
porosityFolder = '95'; % Porosity 95%

% Frequency range
minFreq = 100;   % Minimum frequency in Hz
maxFreq = 4950;  % Maximum frequency in Hz

% Base path where data is stored
basePath = fullfile('/Users/tao', 'Documents', 'MATLAB', 'Fibers');

% Read and Filter Reflection Coefficient Data
for i = 1:length(fiberTypes)
    fiberfolder = fiberTypes{i};
    dataPath = fullfile(basePath, fiberfolder, porosityFolder);
    inputFile = fullfile(dataPath, 'converted_R.txt');
    
    if ~isfile(inputFile)
        error('File not found: %s', inputFile);
    end
    
    % Read the data
    data = readmatrix(inputFile);
    
    % Extract frequency column
    freq = data(:,1); % Frequency in Hz
    
    % Check Number of Columns and Process Data
    numCols = size(data, 2); % Get the number of columns
    window_size = 2; % Define the window size for moving average

    if numCols == 4
        % If 4 columns: real part in column 2, imaginary part in column 4
        R_real = data(:,2);
        R_imag = data(:,4);
        fprintf('Processed as 4-column data: %s\n', inputFile);

    elseif numCols == 3
        % If 3 columns: real part in column 2, imaginary part in column 3
        R_real = data(:,2);
        R_imag = data(:,3);
        fprintf('Processed as 3-column data: %s\n', inputFile);
    
    else
        error('Unexpected number of columns (%d) in file: %s', numCols, inputFile);
    end

    % Filter based on frequency range
    valid_idx = (freq >= minFreq) & (freq <= maxFreq);
    freq_filtered = freq(valid_idx);
    R_real_filtered = movmean(R_real(valid_idx), window_size);
    R_imag_filtered = movmean(R_imag(valid_idx), window_size);

    % Construct complex reflection coefficient
    R_exp_filtered = R_real_filtered + 1i * R_imag_filtered;
    
    % Generate variable names dynamically
    freqVarName = sprintf('%s_95_f', fiberfolder);
    RVarName = sprintf('%s_95_R', fiberfolder);
    
    % Assign variables dynamically
    assignin('base', freqVarName, freq_filtered);
    assignin('base', RVarName, R_exp_filtered);
end

disp('Filtered reflection coefficient data successfully loaded and saved as variables!');

% predicted R and alpha from posterior mean and JCAL model
[Acrylic_95_R_JCAL, ~, Acrylic_95_alpha_JCAL, ~, ~, ~, ~] = jcal_reflection(h, 0.95, data_inferred.Airflow_Resistivity(4), data_inferred.Tortuosity(4), data_inferred.Viscous_Characteristic_Length(4), data_inferred.Thermal_Characteristic_Length(4), data_inferred.Static_Thermal_Permeability(4), Acrylic_95_f, airProperties);
[Silk_95_R_JCAL, ~, Silk_95_alpha_JCAL, ~, ~, ~, ~] = jcal_reflection(h, 0.95, data_inferred.Airflow_Resistivity(12), data_inferred.Tortuosity(12), data_inferred.Viscous_Characteristic_Length(12), data_inferred.Thermal_Characteristic_Length(12), data_inferred.Static_Thermal_Permeability(12), Silk_95_f, airProperties);
[Wool_95_R_JCAL, ~, Wool_95_alpha_JCAL, ~, ~, ~, ~] = jcal_reflection(h, 0.95, data_inferred.Airflow_Resistivity(20), data_inferred.Tortuosity(20), data_inferred.Viscous_Characteristic_Length(20), data_inferred.Thermal_Characteristic_Length(20), data_inferred.Static_Thermal_Permeability(20), Wool_95_f, airProperties);
% predicted R and alpha from simplified JCAL model
[Acrylic_95_R_SJCAL, ~,Acrylic_95_alpha_SJCAL, ~, ~, ~, ~]=jcal_s(1, h, 0.95, coefficients, Acrylic_95_f, airProperties);
[Silk_95_R_SJCAL, ~,Silk_95_alpha_SJCAL, ~, ~, ~, ~]=jcal_s(2, h, 0.95, coefficients, Silk_95_f, airProperties);
[Wool_95_R_SJCAL, ~,Wool_95_alpha_SJCAL, ~, ~, ~, ~]=jcal_s(3, h, 0.95, coefficients, Wool_95_f, airProperties);

%% Figure in manuscript, 95% porosity
% Create a tiled layout
figure;
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Acrylic fiber reflection coefficient
ax1 = nexttile;
s1 = scatter(ax1, Acrylic_95_f, real(Acrylic_95_R), 'filled');
hold on;
p1 = plot(ax1, Acrylic_95_f, real(Acrylic_95_R_JCAL), '--', 'LineWidth', 4);
p2 = plot(ax1, Acrylic_95_f, real(Acrylic_95_R_SJCAL), '-.', 'LineWidth', 4);
s2 = scatter(ax1, Acrylic_95_f, imag(Acrylic_95_R), 'filled');
p3 = plot(ax1, Acrylic_95_f, imag(Acrylic_95_R_JCAL), '--', 'LineWidth', 4);
p4 = plot(ax1, Acrylic_95_f, imag(Acrylic_95_R_SJCAL), '-.', 'LineWidth', 4);
ylabel(ax1, 'Reflection Coefficient');
xticklabels(ax1, {});
title(ax1, 'Acrylic fiber', 'FontSize', 18);

% Silk fiber reflection coefficient
ax2 = nexttile;
scatter(ax2, Silk_95_f, real(Silk_95_R), 'filled');
hold on;
plot(ax2, Silk_95_f, real(Silk_95_R_JCAL), '--', 'LineWidth', 4);
plot(ax2, Silk_95_f, real(Silk_95_R_SJCAL), '-.', 'LineWidth', 4);
scatter(ax2, Silk_95_f, imag(Silk_95_R), 'filled');
plot(ax2, Silk_95_f, imag(Silk_95_R_JCAL), '--', 'LineWidth', 4);
plot(ax2, Silk_95_f, imag(Silk_95_R_SJCAL), '-.', 'LineWidth', 4);
xticklabels(ax2, {});
title(ax2, 'Silk fiber', 'FontSize', 22);

% Wool fiber reflection coefficient
ax3 = nexttile;
scatter(ax3, Wool_95_f, real(Wool_95_R), 'filled');
hold on;
plot(ax3, Wool_95_f, real(Wool_95_R_JCAL), '--', 'LineWidth', 4);
plot(ax3, Wool_95_f, real(Wool_95_R_SJCAL), '-.', 'LineWidth', 4);
scatter(ax3, Wool_95_f, imag(Wool_95_R), 'filled');
plot(ax3, Wool_95_f, imag(Wool_95_R_JCAL), '--', 'LineWidth', 4);
plot(ax3, Wool_95_f, imag(Wool_95_R_SJCAL), '-.', 'LineWidth', 4);
xticklabels(ax3, {});
title(ax3, 'Wool fiber', 'FontSize', 20);

% Acrylic fiber sound absorption coefficient
ax4 = nexttile;
Acrylic_95_alpha = 1 - abs(Acrylic_95_R).^2;
s3 = scatter(ax4, Acrylic_95_f, Acrylic_95_alpha, 'filled');
hold on;
p5 = plot(ax4, Acrylic_95_f, Acrylic_95_alpha_JCAL, '--', 'LineWidth', 4);
p6 = plot(ax4, Acrylic_95_f, Acrylic_95_alpha_SJCAL, '-.', 'LineWidth', 4);
ylabel(ax4, {'Sound Absorption', 'Coefficient (\bf\alpha \rm)'});
xlabel(ax4, 'Frequency (Hz)');

% Silk fiber sound absorption coefficient
ax5 = nexttile;
Silk_95_alpha = 1 - abs(Silk_95_R).^2;
scatter(ax5, Silk_95_f, Silk_95_alpha, 'filled');
hold on;
plot(ax5, Silk_95_f, Silk_95_alpha_JCAL, '--', 'LineWidth', 4);
plot(ax5, Silk_95_f, Silk_95_alpha_SJCAL, '-.', 'LineWidth', 4);
xlabel(ax5, 'Frequency (Hz)', 'FontSize', 30);

% Wool fiber sound absorption coefficient
ax6 = nexttile;
Wool_95_alpha = 1 - abs(Wool_95_R).^2;
scatter(ax6, Wool_95_f, Wool_95_alpha, 'filled');
hold on;
plot(ax6, Wool_95_f, Wool_95_alpha_JCAL, '--', 'LineWidth', 4);
plot(ax6, Wool_95_f, Wool_95_alpha_SJCAL, '-.', 'LineWidth', 4);
xlabel(ax6, 'Frequency (Hz)', 'FontSize', 20);

% Set figure to fullscreen
set(gcf, 'Position', get(0, 'Screensize'));

% Apply formatting to all axes
hAx = findobj(gcf, 'type', 'axes');
for i = 1:numel(hAx)
    set(hAx(i), 'LineWidth', 1, 'FontSize', 30);
    grid(hAx(i), 'on');
    box(hAx(i), 'on');
    set(hAx(i), 'XScale', 'log'); % Set x-axis to log scale
    xlim(hAx(i), [100 6000]); % Set x-axis range
end
hold off;

% Create external legends
lgd1 = legend([s1, p1, p2, s2, p3, p4], ...
    'Real part (Measured)', 'Real part (Original JCAL model)', 'Real part (Simplified JCAL model)', ...
    'Imaginary part (Measured)', 'Imaginary part (Original JCAL model)', 'Imaginary part (Simplified JCAL model)', ...
    'FontSize', 30, 'Location', 'eastoutside');
lgd1.Box = 'off'; % Make legend box invisible
lgd2 = legend([s3, p5, p6], ...
    'Measurement', 'Original JCAL model', 'Simplified JCAL model', ...
    'FontSize', 30, 'Location', 'eastoutside');

% Position the legends outside
lgd1.Layout.Tile = 'east';
lgd2.Layout.Tile = 'east';
lgd2.Box = 'off'; % Make legend box invisible

%% Comparison for 97% Porosity
% Define Fiber Types and Porosity
fiberTypes = {'Acrylic', 'Silk', 'Wool'};
porosityFolder = '97'; % Porosity 97%

% Frequency range
minFreq = 100;   % Minimum frequency in Hz
maxFreq = 4950;  % Maximum frequency in Hz

% Base path where data is stored
basePath = fullfile('/Users/tao', 'Documents', 'MATLAB', 'Fibers');

% Read and Filter Reflection Coefficient Data
for i = 1:length(fiberTypes)
    fiberfolder = fiberTypes{i};
    dataPath = fullfile(basePath, fiberfolder, porosityFolder);
    inputFile = fullfile(dataPath, 'converted_R.txt');
    
    if ~isfile(inputFile)
        error('File not found: %s', inputFile);
    end
    
    % Read the data
    data = readmatrix(inputFile);
    
    % Extract frequency column
    freq = data(:,1); % Frequency in Hz
    
    % Check Number of Columns and Process Data
    numCols = size(data, 2); % Get the number of columns
    window_size = 2; % Define the window size for moving average

    if numCols == 4
        % If 4 columns: real part in column 2, imaginary part in column 4
        R_real = data(:,2);
        R_imag = data(:,4);
        fprintf('Processed as 4-column data: %s\n', inputFile);

    elseif numCols == 3
        % If 3 columns: real part in column 2, imaginary part in column 3
        R_real = data(:,2);
        R_imag = data(:,3);
        fprintf('Processed as 3-column data: %s\n', inputFile);
    
    else
        error('Unexpected number of columns (%d) in file: %s', numCols, inputFile);
    end

    % Filter based on frequency range
    valid_idx = (freq >= minFreq) & (freq <= maxFreq);
    freq_filtered = freq(valid_idx);
    R_real_filtered = movmean(R_real(valid_idx), window_size);
    R_imag_filtered = movmean(R_imag(valid_idx), window_size);

    % Construct complex reflection coefficient
    R_exp_filtered = R_real_filtered + 1i * R_imag_filtered;
    
    % Generate variable names dynamically
    freqVarName = sprintf('%s_97_f', fiberfolder);
    RVarName = sprintf('%s_97_R', fiberfolder);
    
    % Assign variables dynamically
    assignin('base', freqVarName, freq_filtered);
    assignin('base', RVarName, R_exp_filtered);
end

disp('Filtered reflection coefficient data successfully loaded and saved as variables!');

% predicted R and alpha from posterior mean and JCAL model
[Acrylic_97_R_JCAL, ~, Acrylic_97_alpha_JCAL, ~, ~, ~, ~] = jcal_reflection(h, 0.97, data_inferred.Airflow_Resistivity(6), data_inferred.Tortuosity(6), data_inferred.Viscous_Characteristic_Length(6), data_inferred.Thermal_Characteristic_Length(6), data_inferred.Static_Thermal_Permeability(6), Acrylic_97_f, airProperties);
[Silk_97_R_JCAL, ~, Silk_97_alpha_JCAL, ~, ~, ~, ~] = jcal_reflection(h, 0.97, data_inferred.Airflow_Resistivity(14), data_inferred.Tortuosity(14), data_inferred.Viscous_Characteristic_Length(14), data_inferred.Thermal_Characteristic_Length(14), data_inferred.Static_Thermal_Permeability(14), Silk_97_f, airProperties);
[Wool_97_R_JCAL, ~, Wool_97_alpha_JCAL, ~, ~, ~, ~] = jcal_reflection(h, 0.97, data_inferred.Airflow_Resistivity(22), data_inferred.Tortuosity(22), data_inferred.Viscous_Characteristic_Length(22), data_inferred.Thermal_Characteristic_Length(22), data_inferred.Static_Thermal_Permeability(22), Wool_97_f, airProperties);
% predicted R and alpha from simplified JCAL model
[Acrylic_97_R_SJCAL, ~,Acrylic_97_alpha_SJCAL, ~, ~, ~, ~]=jcal_s(1, h, 0.97, coefficients, Acrylic_97_f, airProperties);
[Silk_97_R_SJCAL, ~,Silk_97_alpha_SJCAL, ~, ~, ~, ~]=jcal_s(2, h, 0.97, coefficients, Silk_97_f, airProperties);
[Wool_97_R_SJCAL, ~,Wool_97_alpha_SJCAL, ~, ~, ~, ~]=jcal_s(3, h, 0.97, coefficients, Wool_97_f, airProperties);

%% Figure in manuscript, 97% porosity
% Create a tiled layout
figure;
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Acrylic fiber reflection coefficient
ax1 = nexttile;
s1 = scatter(ax1, Acrylic_97_f, real(Acrylic_97_R), 'filled');
hold on;
p1 = plot(ax1, Acrylic_97_f, real(Acrylic_97_R_JCAL), '--', 'LineWidth', 4);
p2 = plot(ax1, Acrylic_97_f, real(Acrylic_97_R_SJCAL), '-.', 'LineWidth', 4);
s2 = scatter(ax1, Acrylic_97_f, imag(Acrylic_97_R), 'filled');
p3 = plot(ax1, Acrylic_97_f, imag(Acrylic_97_R_JCAL), '--', 'LineWidth', 4);
p4 = plot(ax1, Acrylic_97_f, imag(Acrylic_97_R_SJCAL), '-.', 'LineWidth', 4);
ylabel(ax1, 'Reflection Coefficient');
xticklabels(ax1, {});
title(ax1, 'Acrylic fiber', 'FontSize', 18);

% Silk fiber reflection coefficient
ax2 = nexttile;
scatter(ax2, Silk_97_f, real(Silk_97_R), 'filled');
hold on;
plot(ax2, Silk_97_f, real(Silk_97_R_JCAL), '--', 'LineWidth', 4);
plot(ax2, Silk_97_f, real(Silk_97_R_SJCAL), '-.', 'LineWidth', 4);
scatter(ax2, Silk_97_f, imag(Silk_97_R), 'filled');
plot(ax2, Silk_97_f, imag(Silk_97_R_JCAL), '--', 'LineWidth', 4);
plot(ax2, Silk_97_f, imag(Silk_97_R_SJCAL), '-.', 'LineWidth', 4);
xticklabels(ax2, {});
title(ax2, 'Silk fiber', 'FontSize', 22);

% Wool fiber reflection coefficient
ax3 = nexttile;
scatter(ax3, Wool_97_f, real(Wool_97_R), 'filled');
hold on;
plot(ax3, Wool_97_f, real(Wool_97_R_JCAL), '--', 'LineWidth', 4);
plot(ax3, Wool_97_f, real(Wool_97_R_SJCAL), '-.', 'LineWidth', 4);
scatter(ax3, Wool_97_f, imag(Wool_97_R), 'filled');
plot(ax3, Wool_97_f, imag(Wool_97_R_JCAL), '--', 'LineWidth', 4);
plot(ax3, Wool_97_f, imag(Wool_97_R_SJCAL), '-.', 'LineWidth', 4);
xticklabels(ax3, {});
title(ax3, 'Wool fiber', 'FontSize', 30);

% Acrylic fiber sound absorption coefficient
ax4 = nexttile;
Acrylic_97_alpha = 1 - abs(Acrylic_97_R).^2;
s3 = scatter(ax4, Acrylic_97_f, Acrylic_97_alpha, 'filled');
hold on;
p5 = plot(ax4, Acrylic_97_f, Acrylic_97_alpha_JCAL, '--', 'LineWidth', 4);
p6 = plot(ax4, Acrylic_97_f, Acrylic_97_alpha_SJCAL, '-.', 'LineWidth', 4);
ylabel(ax4, {'Sound Absorption', 'Coefficient (\bf\alpha \rm)'});
xlabel(ax4, 'Frequency (Hz)');

% Silk fiber sound absorption coefficient
ax5 = nexttile;
Silk_97_alpha = 1 - abs(Silk_97_R).^2;
scatter(ax5, Silk_97_f, Silk_97_alpha, 'filled');
hold on;
plot(ax5, Silk_97_f, Silk_97_alpha_JCAL, '--', 'LineWidth', 4);
plot(ax5, Silk_97_f, Silk_97_alpha_SJCAL, '-.', 'LineWidth', 4);
xlabel(ax5, 'Frequency (Hz)', 'FontSize', 30);

% Wool fiber sound absorption coefficient
ax6 = nexttile;
Wool_97_alpha = 1 - abs(Wool_97_R).^2;
scatter(ax6, Wool_97_f, Wool_97_alpha, 'filled');
hold on;
plot(ax6, Wool_97_f, Wool_97_alpha_JCAL, '--', 'LineWidth', 4);
plot(ax6, Wool_97_f, Wool_97_alpha_SJCAL, '-.', 'LineWidth', 4);
xlabel(ax6, 'Frequency (Hz)', 'FontSize', 30);

% Set figure to fullscreen
set(gcf, 'Position', get(0, 'Screensize'));

% Apply formatting to all axes
hAx = findobj(gcf, 'type', 'axes');
for i = 1:numel(hAx)
    set(hAx(i), 'LineWidth', 1, 'FontSize', 40);
    grid(hAx(i), 'on');
    box(hAx(i), 'on');
    set(hAx(i), 'XScale', 'log'); % Set x-axis to log scale
    xlim(hAx(i), [100 6000]); % Set x-axis range
end
hold off;

% Create external legends
lgd1 = legend([s1, p1, p2, s2, p3, p4], ...
    'Real part (measured)', 'Real part (Original JCAL model)', 'Real part (Simplified JCAL model)', ...
    'Imaginary part (measured)', 'Imaginary part (Original JCAL model)', 'Imaginary part (Simplified JCAL model)', ...
    'FontSize', 30, 'Location', 'eastoutside');
lgd1.Box = 'off'; % Make legend box invisible
lgd2 = legend([s3, p5, p6], ...
    'Measurement', 'Original JCAL model', 'Simplified JCAL model', ...
    'FontSize', 30, 'Location', 'eastoutside');
lgd2.Box = 'off'; % Make legend box invisible

% Position the legends outside
lgd1.Layout.Tile = 'east';
lgd2.Layout.Tile = 'east';

%% Comparison for 98% Porosity
% Define Fiber Types and Porosity
fiberTypes = {'Acrylic', 'Silk', 'Wool'};
porosityFolder = '98'; % Porosity 98%

% Frequency range
minFreq = 100;   % Minimum frequency in Hz
maxFreq = 4950;  % Maximum frequency in Hz

% Base path where data is stored
basePath = fullfile('/Users/tao', 'Documents', 'MATLAB', 'Fibers');

% Read and Filter Reflection Coefficient Data
for i = 1:length(fiberTypes)
    fiberfolder = fiberTypes{i};
    dataPath = fullfile(basePath, fiberfolder, porosityFolder);
    inputFile = fullfile(dataPath, 'converted_R.txt');
    
    if ~isfile(inputFile)
        error('File not found: %s', inputFile);
    end
    
    % Read the data
    data = readmatrix(inputFile);
    
    % Extract frequency column
    freq = data(:,1); % Frequency in Hz
    
    % Check Number of Columns and Process Data
    numCols = size(data, 2); % Get the number of columns
    window_size = 2; % Define the window size for moving average

    if numCols == 4
        % If 4 columns: real part in column 2, imaginary part in column 4
        R_real = data(:,2);
        R_imag = data(:,4);
        fprintf('Processed as 4-column data: %s\n', inputFile);

    elseif numCols == 3
        % If 3 columns: real part in column 2, imaginary part in column 3
        R_real = data(:,2);
        R_imag = data(:,3);
        fprintf('Processed as 3-column data: %s\n', inputFile);
    
    else
        error('Unexpected number of columns (%d) in file: %s', numCols, inputFile);
    end

    % Filter based on frequency range
    valid_idx = (freq >= minFreq) & (freq <= maxFreq);
    freq_filtered = freq(valid_idx);
    R_real_filtered = movmean(R_real(valid_idx), window_size);
    R_imag_filtered = movmean(R_imag(valid_idx), window_size);

    % Construct complex reflection coefficient
    R_exp_filtered = R_real_filtered + 1i * R_imag_filtered;
    
    % Generate variable names dynamically
    freqVarName = sprintf('%s_98_f', fiberfolder);
    RVarName = sprintf('%s_98_R', fiberfolder);
    
    % Assign variables dynamically
    assignin('base', freqVarName, freq_filtered);
    assignin('base', RVarName, R_exp_filtered);
end

disp('Filtered reflection coefficient data successfully loaded and saved as variables!');

% predicted R and alpha from posterior mean and JCAL model
[Acrylic_98_R_JCAL, ~, Acrylic_98_alpha_JCAL, ~, ~, ~, ~] = jcal_reflection(h, 0.98, data_inferred.Airflow_Resistivity(7), data_inferred.Tortuosity(7), data_inferred.Viscous_Characteristic_Length(7), data_inferred.Thermal_Characteristic_Length(7), data_inferred.Static_Thermal_Permeability(7), Acrylic_98_f, airProperties);
[Silk_98_R_JCAL, ~, Silk_98_alpha_JCAL, ~, ~, ~, ~] = jcal_reflection(h, 0.98, data_inferred.Airflow_Resistivity(15), data_inferred.Tortuosity(15), data_inferred.Viscous_Characteristic_Length(15), data_inferred.Thermal_Characteristic_Length(15), data_inferred.Static_Thermal_Permeability(15), Silk_98_f, airProperties);
[Wool_98_R_JCAL, ~, Wool_98_alpha_JCAL, ~, ~, ~, ~] = jcal_reflection(h, 0.98, data_inferred.Airflow_Resistivity(23), data_inferred.Tortuosity(23), data_inferred.Viscous_Characteristic_Length(23), data_inferred.Thermal_Characteristic_Length(23), data_inferred.Static_Thermal_Permeability(23), Wool_98_f, airProperties);
% predicted R and alpha from simplified JCAL model
[Acrylic_98_R_SJCAL, ~,Acrylic_98_alpha_SJCAL, ~, ~, ~, ~]=jcal_s(1, h, 0.98, coefficients, Acrylic_98_f, airProperties);
[Silk_98_R_SJCAL, ~,Silk_98_alpha_SJCAL, ~, ~, ~, ~]=jcal_s(2, h, 0.98, coefficients, Silk_98_f, airProperties);
[Wool_98_R_SJCAL, ~,Wool_98_alpha_SJCAL, ~, ~, ~, ~]=jcal_s(3, h, 0.98, coefficients, Wool_98_f, airProperties);

%% Figure in manuscript, 98% porosity
% Create a tiled layout
figure;
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Acrylic fiber reflection coefficient
ax1 = nexttile;
s1 = scatter(ax1, Acrylic_98_f, real(Acrylic_98_R), 'filled');
hold on;
p1 = plot(ax1, Acrylic_98_f, real(Acrylic_98_R_JCAL), '--', 'LineWidth', 4);
p2 = plot(ax1, Acrylic_98_f, real(Acrylic_98_R_SJCAL), '-.', 'LineWidth', 4);
s2 = scatter(ax1, Acrylic_98_f, imag(Acrylic_98_R), 'filled');
p3 = plot(ax1, Acrylic_98_f, imag(Acrylic_98_R_JCAL), '--', 'LineWidth', 4);
p4 = plot(ax1, Acrylic_98_f, imag(Acrylic_98_R_SJCAL), '-.', 'LineWidth', 4);
ylabel(ax1, 'Reflection Coefficient');
xticklabels(ax1, {});
title(ax1, 'Acrylic fiber', 'FontSize', 18);

% Silk fiber reflection coefficient
ax2 = nexttile;
scatter(ax2, Silk_98_f, real(Silk_98_R), 'filled');
hold on;
plot(ax2, Silk_98_f, real(Silk_98_R_JCAL), '--', 'LineWidth', 4);
plot(ax2, Silk_98_f, real(Silk_98_R_SJCAL), '-.', 'LineWidth', 4);
scatter(ax2, Silk_98_f, imag(Silk_98_R), 'filled');
plot(ax2, Silk_98_f, imag(Silk_98_R_JCAL), '--', 'LineWidth', 4);
plot(ax2, Silk_98_f, imag(Silk_98_R_SJCAL), '-.', 'LineWidth', 4);
xticklabels(ax2, {});
title(ax2, 'Silk fiber', 'FontSize', 22);

% Wool fiber reflection coefficient
ax3 = nexttile;
scatter(ax3, Wool_98_f, real(Wool_98_R), 'filled');
hold on;
plot(ax3, Wool_98_f, real(Wool_98_R_JCAL), '--', 'LineWidth', 4);
plot(ax3, Wool_98_f, real(Wool_98_R_SJCAL), '-.', 'LineWidth', 4);
scatter(ax3, Wool_98_f, imag(Wool_98_R), 'filled');
plot(ax3, Wool_98_f, imag(Wool_98_R_JCAL), '--', 'LineWidth', 4);
plot(ax3, Wool_98_f, imag(Wool_98_R_SJCAL), '-.', 'LineWidth', 4);
xticklabels(ax3, {});
title(ax3, 'Wool fiber', 'FontSize', 30);

% Acrylic fiber sound absorption coefficient
ax4 = nexttile;
Acrylic_98_alpha = 1 - abs(Acrylic_98_R).^2;
s3 = scatter(ax4, Acrylic_98_f, Acrylic_98_alpha, 'filled');
hold on;
p5 = plot(ax4, Acrylic_98_f, Acrylic_98_alpha_JCAL, '--', 'LineWidth', 4);
p6 = plot(ax4, Acrylic_98_f, Acrylic_98_alpha_SJCAL, '-.', 'LineWidth', 4);
ylabel(ax4, {'Sound Absorption', 'Coefficient (\bf\alpha \rm)'});
xlabel(ax4, 'Frequency (Hz)');

% Silk fiber sound absorption coefficient
ax5 = nexttile;
Silk_98_alpha = 1 - abs(Silk_98_R).^2;
scatter(ax5, Silk_98_f, Silk_98_alpha, 'filled');
hold on;
plot(ax5, Silk_98_f, Silk_98_alpha_JCAL, '--', 'LineWidth', 4);
plot(ax5, Silk_98_f, Silk_98_alpha_SJCAL, '-.', 'LineWidth', 4);
xlabel(ax5, 'Frequency (Hz)', 'FontSize', 30);

% Wool fiber sound absorption coefficient
ax6 = nexttile;
Wool_98_alpha = 1 - abs(Wool_98_R).^2;
scatter(ax6, Wool_98_f, Wool_98_alpha, 'filled');
hold on;
plot(ax6, Wool_98_f, Wool_98_alpha_JCAL, '--', 'LineWidth', 4);
plot(ax6, Wool_98_f, Wool_98_alpha_SJCAL, '-.', 'LineWidth', 4);
xlabel(ax6, 'Frequency (Hz)', 'FontSize', 30);

% Set figure to fullscreen
set(gcf, 'Position', get(0, 'Screensize'));

% Apply formatting to all axes
hAx = findobj(gcf, 'type', 'axes');
for i = 1:numel(hAx)
    set(hAx(i), 'LineWidth', 1, 'FontSize', 40);
    grid(hAx(i), 'on');
    box(hAx(i), 'on');
    set(hAx(i), 'XScale', 'log'); % Set x-axis to log scale
    xlim(hAx(i), [100 6000]); % Set x-axis range
end
hold off;

% Create external legends
lgd1 = legend([s1, p1, p2, s2, p3, p4], ...
    'Real part (measured)', 'Real part (Original JCAL model)', 'Real part (Simplified JCAL model)', ...
    'Imaginary part (measured)', 'Imaginary part (Original JCAL model)', 'Imaginary part (Simplified JCAL model)', ...
    'FontSize', 30, 'Location', 'eastoutside');
lgd1.Box = 'off'; % Make legend box invisible
lgd2 = legend([s3, p5, p6], ...
    'Measurement', 'Original JCAL model', 'Simplified JCAL model', ...
    'FontSize', 30, 'Location', 'eastoutside');
box off;

% Position the legends outside
lgd1.Layout.Tile = 'east';
lgd2.Layout.Tile = 'east';
lgd2.Box = 'off'; % Make legend box invisible

%% Comaprison of all porosities
% Define Fiber Type (Acrylic, Silk, or Wool)
fiberType = 'Wool'; % Example: 'Acrylic', 'Silk', 'Wool'

% Debugging: Check the value of fiberType
disp(['fiberType before switch: ', fiberType]);

switch fiberType
    case 'Acrylic'
        fiberTypeIndex = 1;
        coefficients = struct('a1', a1, 'b1', b1, 'c1', c1, 'd1', d1, 'e1', e1);
    case 'Silk'
        fiberTypeIndex = 2;
        coefficients = struct('a2', a2, 'b2', b2, 'c2', c2, 'd2', d2, 'e2', e2);
    case 'Wool'
        fiberTypeIndex = 3;
        coefficients = struct('a3', a3, 'b3', b3, 'c3', c3, 'd3', d3, 'e3', e3);
    otherwise
        error('Invalid fiberType. Allowed values: Acrylic, Silk, Wool.');
end

% Porosity range and frequency range
porosityValues = 0.92:0.01:0.99; % Porosity range from table
minFreq = 100; maxFreq = 4950; % Frequency range

% Base path where data is stored
basePath = fullfile('/Users/tao', 'Documents', 'MATLAB', 'Fibers', fiberType);

% Read and Filter Reflection Coefficient Data
numPorosities = length(porosityValues);
freqData = cell(1, numPorosities);
R_expData = cell(1, numPorosities);

for i = 1:numPorosities
    porosityFolder = num2str(round(porosityValues(i) * 100)); % Convert to folder name
    dataPath = fullfile(basePath, porosityFolder);
    inputFile = fullfile(dataPath, 'converted_R.txt');

    if ~isfile(inputFile)
        warning('File not found: %s', inputFile);
        continue;
    end
    
    % Read the data
    data = readmatrix(inputFile);
    freq = data(:, 1); % Frequency column
    numCols = size(data, 2); % Number of columns in the data
    window_size = 2; % Moving average filter

    % Process data based on column count
    if numCols == 4
        R_real = data(:, 2); % Real part
        R_imag = data(:, 4); % Imaginary part
    elseif numCols == 3
        R_real = data(:, 2); % Real part
        R_imag = data(:, 3); % Imaginary part
    else
        error('Unexpected number of columns (%d) in file: %s', numCols, inputFile);
    end

    % Filter based on frequency range
    valid_idx = (freq >= minFreq) & (freq <= maxFreq);
    freq_filtered = freq(valid_idx);
    R_real_filtered = movmean(R_real(valid_idx), window_size); % Apply moving average filter
    R_imag_filtered = movmean(R_imag(valid_idx), window_size); % Apply moving average filter
    
    % Construct complex reflection coefficient
    R_exp_filtered = R_real_filtered + 1i * R_imag_filtered;
    
    % Store filtered data in cell arrays
    freqData{i} = freq_filtered;
    R_expData{i} = R_exp_filtered;
end

disp('Filtered reflection coefficient data successfully loaded.');

% Read Extracted Parameters from File
paramsFile = 'Extracted_All_Params.txt'; % Update if necessary
paramsTable = readtable(paramsFile, 'PreserveVariableNames', true);

% Check bounds and extract parameters for selected fiber type (e.g., Acrylic)
switch fiberTypeIndex
    case 1
        fiberParams = paramsTable(1:8, :); % Extract rows for Acrylic
    case 2
        fiberParams = paramsTable(9:16, :); % Extract rows for Silk
    case 3
        fiberParams = paramsTable(17:24, :); % Extract rows for Wool
    otherwise
        error('Invalid fiber type specified');
end


% Extract Porosity and Parameters (ignoring SDs)
phi = fiberParams.phi; % Porosity values
inferred_params = [fiberParams.Airflow_Resistivity, fiberParams.Tortuosity, ...
                   fiberParams.Viscous_Characteristic_Length, fiberParams.Thermal_Characteristic_Length, ...
                   fiberParams.Static_Thermal_Permeability];

disp(['Extracted parameters for fiber type ', num2str(fiberType), '.']);

% Create a tiled layout
figure;
tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Store legend handles for later use
legendHandles = [];

% Ensure the number of porosities doesn't exceed the available parameters
numPorosities = min(numPorosities, size(inferred_params, 1)); % Adjust to the number of available parameter sets

for i = 1:numPorosities
    ax = nexttile;

    % Extract Experimental Data (Measured Reflection Coefficient)
    R_measured = R_expData{i};  % Complex reflection coefficient
    freq_current = freqData{i}; % Corresponding frequency values

    % Extract Parameters
    sigma_i = inferred_params(i, 1);
    alpha_inf_i = inferred_params(i, 2);
    lambda_i = inferred_params(i, 3);
    lambda_prime_i = inferred_params(i, 4);
    k0_prime_i = inferred_params(i, 5);

    % Compute Reflection Coefficient using Original JCAL Model
    [Reflect_JCAL, ~, ~, ~, ~, ~, ~] = jcal_reflection(h, phi(i), sigma_i, alpha_inf_i, lambda_i, lambda_prime_i, k0_prime_i, freq_current, airProperties);

    % Compute Reflection Coefficient using Simplified JCAL Model
    [Reflect_SJCAL, ~, ~, ~, ~, ~, ~] = jcal_s(fiberTypeIndex, h, phi(i), coefficients, freq_current, airProperties);

    % Plot Real Part
    s1 = scatter(ax, freq_current, real(R_measured), 'filled', 'DisplayName', 'Real part (measured)');
    hold on;
    p1 = plot(ax, freq_current, real(Reflect_JCAL), '--', 'LineWidth', 4, 'DisplayName', 'Real part (Original JCAL model)');
    p2 = plot(ax, freq_current, real(Reflect_SJCAL), '-.', 'LineWidth', 4, 'DisplayName', 'Real part (Simplified JCAL model)');

    % Plot Imaginary Part
    s2 = scatter(ax, freq_current, imag(R_measured), 'filled', 'DisplayName', 'Imaginary part (measured)');
    p3 = plot(ax, freq_current, imag(Reflect_JCAL), '--', 'LineWidth', 4, 'DisplayName', 'Imaginary part (Original JCAL model)');
    p4 = plot(ax, freq_current, imag(Reflect_SJCAL), '-.', 'LineWidth', 4, 'DisplayName', 'Imaginary part (Simplified JCAL model)');

    % Store legend handles (only once for clarity)
    if i == 1
        legendHandles = [s1, p1, p2, s2, p3, p4];
    end

    % Formatting
    title(ax, sprintf('Porosity: %.0f%%', porosityValues(i) * 100), 'FontSize', 18);
    set(ax, 'XScale', 'log', 'LineWidth', 1, 'FontSize', 30);
    xlim(ax, [100 6000]);
    grid(ax, 'on');
    box(ax, 'on');

    % Labels
    if i == 1 || i == 5
        ylabel(ax, 'Reflection Coefficient');
    end
    if i >= 5
        xlabel(ax, 'Frequency (Hz)');
    end
end

% Set figure to fullscreen
set(gcf, 'Position', get(0, 'Screensize'));

% Create external legend
lgd = legend(legendHandles, ...
    'Real part (Measured)', 'Real part (Original JCAL model)', 'Real part (Simplified JCAL model)', ...
    'Imaginary part (Measured)', 'Imaginary part (Original JCAL model)', 'Imaginary part (Simplified JCAL model)', ...
    'FontSize', 30, 'Location', 'eastoutside');
lgd.Box = 'off'; % Make legend box invisible
lgd.Layout.Tile = 'east';

disp('Plotting complete.');


% Create a tiled layout
figure;
t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Store legend handles for later use
legendHandles = [];

% Ensure the number of porosities doesn't exceed the available parameters
numPorosities = min(numPorosities, size(inferred_params, 1)); % Adjust to the number of available parameter sets

for i = 1:numPorosities
    ax = nexttile;

    % Extract Experimental Data (Measured Reflection Coefficient)
    R_measured = R_expData{i};  % Complex reflection coefficient
    freq_current = freqData{i}; % Corresponding frequency values

    % Compute Measured Sound Absorption
    alpha_measured = 1 - abs(R_measured).^2;  % Sound absorption = 1 - |R_measured|^2

    % Extract Parameters
    sigma_i = inferred_params(i, 1);
    alpha_inf_i = inferred_params(i, 2);
    lambda_i = inferred_params(i, 3);
    lambda_prime_i = inferred_params(i, 4);
    k0_prime_i = inferred_params(i, 5);

    % Compute Sound Absorption (alpha) using Original JCAL Model
    [~, ~, alpha_JCAL, ~, ~, ~, ~] = jcal_reflection(h, phi(i), sigma_i, alpha_inf_i, lambda_i, lambda_prime_i, k0_prime_i, freq_current, airProperties);

    % Compute Sound Absorption (alpha) using Simplified JCAL Model
    [~, ~, alpha_SJCAL, ~, ~, ~, ~] = jcal_s(fiberTypeIndex, h, phi(i), coefficients, freq_current, airProperties);

    % Plot Sound Absorption (alpha)
    p1 = plot(ax, freq_current, alpha_measured, '-', 'LineWidth', 4, 'DisplayName', 'Sound absorption (Measured)');
    hold on;
    p2 = plot(ax, freq_current, alpha_JCAL, '--', 'LineWidth', 4, 'DisplayName', 'Sound absorption (Original JCAL model)');
    p3 = plot(ax, freq_current, alpha_SJCAL, '-.', 'LineWidth', 4, 'DisplayName', 'Sound absorption (Simplified JCAL model)');

    % Formatting
    title(ax, sprintf('Porosity: %.0f%%', porosityValues(i) * 100), 'FontSize', 18);
    set(ax, 'XScale', 'log', 'LineWidth', 1, 'FontSize', 30);
    xlim(ax, [100 6000]);
    grid(ax, 'on');
    box(ax, 'on');

    % Labels
    if i == 1 || i == 5
        ylabel(ax, 'Sound Absorption Coefficient (\bf\alpha \rm)');
    end
    if i >= 5
        xlabel(ax, 'Frequency (Hz)');
    end
end

% Set figure to fullscreen
set(gcf, 'Position', get(0, 'Screensize'));

% Create external legend
lgd = legend([p1, p2, p3], ...
    'Measured', 'Original JCAL model', 'Simplified JCAL model', ...
    'FontSize', 30, 'Location', 'eastoutside');
lgd.Box = 'off'; % Make legend box invisible
lgd.Layout.Tile = 'east';

disp('Plotting complete.');


