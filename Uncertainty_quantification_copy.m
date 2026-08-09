%% 1️ - Set Fiber Type, Porosity, and Variable
fiberfolder = 'Acrylic';  % Change to 'Silk', 'Wool', etc.
porosityfolder = '94';    % Change to 93, 94, ..., 99

% Convert porosity folder to actual porosity (e.g., '92' -> 0.92)
phi = str2double(porosityfolder) / 100;

variable_of_interest = 'R'; % Reflection coefficient

% Define fiber parameters
fiberParams = struct();
fiberParams.fiberfolder = fiberfolder;  % Fiber type (e.g., 'Acrylic')
fiberParams.porosityfolder = porosityfolder;  % Porosity folder (e.g., '92')
fiberParams.phi = phi;  % Porosity (e.g., 0.92)

%% 2️ - Define File Paths
basePath = fullfile('/Users/tao', 'Documents', 'MATLAB', 'Fibers'); % Full base path
dataPath = fullfile(basePath, fiberfolder, porosityfolder); % Folder where R.txt is located
inputFile = fullfile(dataPath, 'R.txt'); % We only process R.txt
outputFile = fullfile(dataPath, 'converted_R.txt'); % Save converted file

%% 3 - Check if Folder Contains R.txt
disp(['Checking files in folder: ', dataPath]);

filesInFolder = dir(dataPath);  % Get a list of all files in the folder
fileNames = {filesInFolder.name}; % Extract file names

disp('Files in the folder:');
disp(fileNames); % Show all files in the folder

if ~ismember('R.txt', fileNames) % Check if R.txt is present
    error('R.txt not found in %s. Available files are: %s', dataPath, strjoin(fileNames, ', '));
end

disp(['Processing file: ', inputFile]);

%% 4️ - Read & Convert Comma Decimals to Dot Format
fid = fopen(inputFile, 'r'); % Open original file
if fid == -1
    error('Cannot open file: %s', inputFile);
end
rawData = fread(fid, '*char')'; % Read as string
fclose(fid);

cleanedData = strrep(rawData, ',', '.'); % Replace commas with dots

%% 5️ - Save the Converted Data
fid = fopen(outputFile, 'w'); % Open new file for writing
if fid == -1
    error('Cannot create file: %s', outputFile);
end
fwrite(fid, cleanedData);
fclose(fid);

disp(['Converted file saved as: ', outputFile]);

%% 6️ - Load the Converted Data
% Load corrected data
data = readmatrix(outputFile); 

% Extract frequency and complex reflection coefficient components
freq = data(:,1); % Frequency in Hz

% Define frequency limits
mini_freq = 100;  % Minimum frequency in Hz
max_freq = 4950;  % Maximum frequency in Hz

% Define the window size for the moving average
window_size = 2;  % Adjust depending on data characteristics

% Filter data: Keep only frequencies between mini_freq and max_freq
valid_idx = (freq >= mini_freq) & (freq <= max_freq);
freq = freq(valid_idx); % Filtered frequencies

% Check Number of Columns and Process Data
numCols = size(data, 2); % Get the number of columns

if numCols == 4
    % If 4 columns: real part in column 2, imaginary part in column 4
    R_real_filtered = movmean(data(valid_idx,2), window_size); 
    R_imag_filtered = movmean(data(valid_idx,4), window_size); 
    fprintf('Processed as 4-column data.\n');

elseif numCols == 3
    % If 3 columns: real part in column 2, imaginary part in column 3
    R_real_filtered = movmean(data(valid_idx,2), window_size); 
    R_imag_filtered = movmean(data(valid_idx,3), window_size); 
    fprintf('Processed as 3-column data.\n');

else
    error('Unexpected number of columns (%d) in file: %s', numCols, outputFile);
end

% Construct complex reflection coefficient
R_exp = R_real_filtered + 1i * R_imag_filtered;

% Concatenate real and imaginary parts into a single column vector
R_exp_concat = [real(R_exp); imag(R_exp)];

disp('Data successfully loaded and filtered!');

%% 7 - Example: Call getFluidProperties function to retrieve fluid properties
% Retrieve fluid properties from the getFluidProperties function
[thickness, temperature_K, pressure, rel_humidity, density_humid_air, Cp, Cv, eta, gamma, c, kappa, Pr] = getFluidProperties(fiberfolder, porosityfolder);

h = thickness * 1e-3; % Convert thickness to meters (1 mm = 1e-3 m)

% Display retrieved fluid properties
disp(['Sample thickness: ', num2str(thickness), ' mm']);
disp(['Temperature: ', num2str(temperature_K), ' K']);
disp(['Pressure: ', num2str(pressure), ' Pa']); % Display pressure in Pascals
disp(['Relative Humidity: ', num2str(rel_humidity)]);
disp(['Density of humid air: ', num2str(density_humid_air), ' kg/m^3']);
disp(['Cp (Specific Heat at Constant Pressure): ', num2str(Cp), ' J/kg K']);
disp(['Cv (Specific Heat at Constant Volume): ', num2str(Cv), ' J/kg K']);
disp(['Dynamic viscosity: ', num2str(eta), ' Pa.s']);
disp(['Gamma (ratio of specific heats): ', num2str(gamma)]);
disp(['Speed of sound: ', num2str(c), ' m/s']);
disp(['Thermal conductivity: ', num2str(kappa), ' W/m K']);
disp(['Prandtl number: ', num2str(Pr)]);

% Create and populate the airProperties structure with relevant data
airProperties = struct();

% Retrieve fluid properties and store them in the structure (again as per your logic)
[~, ~, pressure, ~, density_humid_air, ~, ~, eta, gamma, c, ~, Pr] = getFluidProperties(fiberfolder, porosityfolder);

% Populate the airProperties structure with fluid property values
airProperties.density_humid_air = density_humid_air;
airProperties.speed_of_sound = c;
airProperties.impedance = density_humid_air * c;  % Impedance of air (rho * c)
airProperties.eta = eta;  % Dynamic viscosity (Pa.s)
airProperties.gamma = gamma;  % Ratio of specific heats (Cp/Cv)
airProperties.Pr = Pr;  % Prandtl number
airProperties.pressure = pressure;  % Air pressure (Pa)

% Define parameters
sigma = 5000; % Example value for sigma, adjust accordingly
alpha_infin = 1; % Example value for alpha_infin
lambda = 1e-4; % Example value for lambda
lambda_prime = 2e-4; % Example value for lambda_prime
k0_prime = 1e-8; % Example value for k0_prime

% Pass the structure to jcal_reflection
[Reflect, Zs_norm, alpha, Re, Im, Zc, ~] = jcal_reflection(h, phi, sigma, alpha_infin, lambda, lambda_prime, k0_prime, freq, airProperties);

%% 8 - Set Bayesian Inference Parameters
% Get constraints from getFiberConstraints function
[lb, ub] = getFiberConstraints(fiberfolder, phi, airProperties);

% Step size (now 6 values, including for sigma_noise)
step_size = [50, 0.001, 1e-7, 1e-7, 1e-9, 0.001];

n_samples = 1000000;  % Number of MCMC iterations
n_chains = 3;        % Number of chains

% Initialize each chain randomly within the full parameter bounds
init_params_all = zeros(n_chains, length(lb));

for chain = 1:n_chains
    init_params_all(chain, :) = lb + (ub - lb) .* rand(1, length(lb));
end

%% 9 - Run Bayesian MCMC Sampling with Multiple Chains
% Run multiple chains to calculate R-hat diagnostics
all_samples = zeros(n_chains, n_samples, length(init_params)); 

for chain = 1:n_chains
    all_samples(chain, :, :) = metropolis_hastings( ...
        n_samples, init_params_all(chain, :), h, freq, R_exp_concat, ...
        step_size, fiberfolder, phi, airProperties);
end

% Compute R-hat diagnostics
r_hat = zeros(1, length(init_params)); 
for param_idx = 1:length(init_params)
    % Extract samples of this parameter from all chains
    chain_samples = squeeze(all_samples(:, :, param_idx));  % (n_chains x n_samples)

    % Per-chain means
    chain_means = mean(chain_samples, 2);
    overall_mean = mean(chain_means);

    % Within-chain variance W
    W = mean(var(chain_samples, 0, 2));  % Variance for each chain, then averaged

    % Between-chain variance B
    B = n_samples * var(chain_means, 1);  % Use population variance (normalize by m)

    % Estimated posterior variance V_hat
    V_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B;

    % Gelman-Rubin R-hat
    r_hat(param_idx) = sqrt(V_hat / W);
end

% Display R-hat values
disp('R-hat Diagnostics:');
disp(r_hat);


%% 10 - Check Convergence
% Ideally, R-hat should be close to 1.0 for good convergence
if all(r_hat < 1.1)
    disp('MCMC convergence appears good (R-hat < 1.1)');
else
    disp('MCMC may not have converged (R-hat > 1.1)');
end

%% 11 - Run Bayesian MCMC Sampling (Final Chain)
%% 11 - Posterior Sampling: Combine Chains After Burn-in
burn_in = 150000;  % Number of burn-in iterations
samples_per_chain = n_samples - burn_in;
n_params = size(all_samples, 3);

% Preallocate matrix for combined post-burn-in samples
samples_after_burnin = zeros(n_chains * samples_per_chain, n_params);

% Collect post-burn-in samples from all chains
row_start = 1;
for chain = 1:n_chains
    chain_samples = squeeze(all_samples(chain, burn_in+1:end, :));  % (samples_per_chain x n_params)
    row_end = row_start + samples_per_chain - 1;
    samples_after_burnin(row_start:row_end, :) = chain_samples;
    row_start = row_end + 1;
end

% Compute posterior mean and standard deviation
estimated_params = mean(samples_after_burnin, 1);
param_std = std(samples_after_burnin, 0, 1);

% Compute predicted reflection coefficient using estimated parameters
[R_predicted, ~, ~, ~, ~, ~, ~] = jcal_reflection(h, phi, ...
    estimated_params(1), estimated_params(2), estimated_params(3), ...
    estimated_params(4), estimated_params(5), freq, airProperties);

%% 12 - Visualization
% Compute Sound Absorption Coefficient (alpha) for comparison
alpha_exp = 1 - abs(R_exp).^2;
alpha_predicted = 1 - abs(R_predicted).^2;

% Reflection Coefficients and Sound Absorption using JCAL_O
[Zs_norm_O, Reflect_O, alpha_O, Re_O, Im_O, Zc_O, k_O] = JCAL_O(freq, h, phi, ...
    estimated_params(1), estimated_params(2), estimated_params(3), ...
    estimated_params(4), estimated_params(5));

% Parameter names and bounds for display
param_names = {'Airflow Resistivity', 'Tortuosity', ...
               'Viscous Characteristic Length', 'Thermal Characteristic Length', ...
               'Static Thermal Permeability', 'Sigma Noise'};

param_constraints = [lb; ub];  % Combine bounds

% Display inferred parameter values and standard deviations
disp('Inferred Parameter Values (Posterior Means):');
for i = 1:length(param_names)
    fprintf('%s: %.6e ± %.6e\n', param_names{i}, estimated_params(i), param_std(i));
end

disp('Parameter Constraints (Lower Bound, Upper Bound):');
for i = 1:length(param_names)
    fprintf('%s: [%.6e, %.6e]\n', param_names{i}, param_constraints(1, i), param_constraints(2, i));
end

% Trace Plots for All Chains and Parameters
figure;
tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for param_idx = 1:n_params
    nexttile;
    hold on;

    % Plot all chains
    for chain = 1:n_chains
        param_trace = squeeze(all_samples(chain, :, param_idx));
        if param_idx == 6
            param_trace = param_trace * 100;  % Convert sigma_noise to %
        end
        plot(param_trace, 'DisplayName', ['Chain ', num2str(chain)]);
    end

    % Add bounds
    if param_idx == 6
        yline(param_constraints(1, param_idx) * 100, '--r', 'LineWidth', 2, 'DisplayName', 'Lower Bound');
        yline(param_constraints(2, param_idx) * 100, '--r', 'LineWidth', 2, 'DisplayName', 'Upper Bound');
        ylabel('Value (%)');
    else
        yline(param_constraints(1, param_idx), '--r', 'LineWidth', 2, 'DisplayName', 'Lower Bound');
        yline(param_constraints(2, param_idx), '--r', 'LineWidth', 2, 'DisplayName', 'Upper Bound');
        ylabel('Value');
    end

    % Add burn-in marker
    xline(burn_in, '-.k', 'LineWidth', 2, 'DisplayName', 'Burn-in');

    title(['Trace - ', param_names{param_idx}]);
    xlabel('Iteration');
    legend('Location', 'northeast');
    hold off;
end

sgtitle('Trace Plots of MCMC Samples');

% Style: Make axes clean and large
set(gcf, 'Position', get(0, 'Screensize'));
hAx = findobj(gcf, 'type', 'axes');
for i = 1:numel(hAx)
    set(hAx(i), 'LineWidth', 1, 'FontSize', 20);
    grid(hAx(i), 'on');
    box(hAx(i), 'on');
end


%% Histograms of Estimated Parameters
figure;
tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:6
    nexttile;
    histogram(samples_after_burnin(:, i), 50, 'FaceColor', 'blue', 'EdgeColor', 'black');
    hold on;
    xline(estimated_params(i), '-k', 'LineWidth', 2); % Mean
    xline(param_constraints(1, i), '--r', 'LineWidth', 2); % Lower Bound
    xline(param_constraints(2, i), '--r', 'LineWidth', 2); % Upper Bound
    hold off;
    title(['Histogram - ', param_names{i}]);
    xlabel('Value');
    ylabel('Frequency');
    legend('Location', 'northwest');
end
sgtitle('Histograms of Estimated Parameters');

% Apply style modifications using a loop
set(gcf, 'Position', get(0, 'Screensize'));  % Maximize figure window
hAx = findobj(gcf, 'type', 'axes');
for i = 1:numel(hAx)
    set(hAx(i), 'LineWidth', 1, 'FontSize', 20);
    grid(hAx(i), 'on');
    box(hAx(i), 'on');
end

%% Standard Deviation Plot
figure;
bar(param_std./estimated_params, 'FaceColor', 'c', 'EdgeColor', 'k');
set(gca, 'XTickLabel', param_names, 'XTickLabelRotation', 30);
ylabel('Standard Deviation');
title('Standard Deviation/Mean Value of Estimated Parameters');
grid on;

%% Reflection & Sound Absorption Coefficient Comparison
figure;
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Real Part of Reflection Coefficient
nexttile;
scatter(freq, real(R_exp), 'red', '.', 'DisplayName', 'Measured (Real)');
hold on;
plot(freq, real(R_predicted), 'b-', 'LineWidth', 2, 'DisplayName', 'JCAL Predicted (Real)');
plot(freq, real(Reflect_O), '--', 'LineWidth', 2, 'DisplayName', 'Original JCAL (Real)');
xlabel('Frequency (Hz)');
ylabel('Real Part of R');
legend;
grid on;
title('Real Reflection Coefficient (R)');

% Imaginary Part of Reflection Coefficient
nexttile;
scatter(freq, imag(R_exp), 'red', '.', 'DisplayName', 'Measured (Imaginary)');
hold on;
plot(freq, imag(R_predicted), 'b-', 'LineWidth', 2, 'DisplayName', 'JCAL Predicted (Imaginary)');
plot(freq, imag(Reflect_O), '--', 'LineWidth', 2, 'DisplayName', 'Original JCAL (Imaginary)');
xlabel('Frequency (Hz)');
ylabel('Imaginary Part of R');
legend('Location', 'northwest');
grid on;
title('Imaginary Reflection Coefficient (R)');

% Sound Absorption Coefficient
nexttile([1 2]); % Span across two tiles
scatter(freq, alpha_exp, 'red', '.', 'DisplayName', 'Measured');
hold on;
plot(freq, alpha_predicted, 'b-', 'LineWidth', 2, 'DisplayName', 'JCAL Predicted');
plot(freq, alpha_O, '--', 'LineWidth', 2, 'DisplayName', 'Original JCAL');
xlabel('Frequency (Hz)');
ylabel('Sound Absorption Coefficient');
legend('Location', 'northwest');
grid on;
title('Experimental vs. Predicted Sound Absorption Coefficient');

% Apply style modifications using a loop
set(gcf, 'Position', get(0, 'Screensize'));  % Maximize figure window
hAx = findobj(gcf, 'type', 'axes');
for i = 1:numel(hAx)
    set(hAx(i), 'LineWidth', 1, 'FontSize', 30);
    grid(hAx(i), 'on');
    box(hAx(i), 'on');
    set(hAx(i), 'XScale', 'log'); % Set x-axis to log scale
end

%% Number of posterior predictive samples
num_posterior_samples = 20000;

% Define the filename based on fiberfolder and porosityfolder
filename = sprintf('%s_%s_random_params.txt', fiberfolder, porosityfolder);

% Randomly select 1000 parameter sets from the posterior samples (after burn-in)
random_indices = randperm(size(samples_after_burnin, 1), num_posterior_samples);
sampled_params = samples_after_burnin(random_indices, :);

% Save Data to File
fileID = fopen(filename, 'w');

% Check if file opened successfully
if fileID == -1
    error('Could not open file for writing.');
end

% Write Header
fprintf(fileID, 'Randomly Selected Posterior Samples\n');
fprintf(fileID, 'Fiber Type: %s\n', fiberfolder);
fprintf(fileID, 'Porosity: %s\n\n', porosityfolder);

% Print Column Headers
num_params = size(sampled_params, 2);
param_names = arrayfun(@(x) sprintf('Param_%d', x), 1:num_params, 'UniformOutput', false);
fprintf(fileID, '%-15s', 'Sample Index');
fprintf(fileID, '%-15s', param_names{:});
fprintf(fileID, '\n');

% Write Parameter Values
for i = 1:num_posterior_samples
    fprintf(fileID, '%-15d', i); % Sample index
    fprintf(fileID, '%.6e ', sampled_params(i, :)); % Parameter values
    fprintf(fileID, '\n');
end

% Close the file
fclose(fileID);

% Confirm saving
fprintf('Saved randomly selected parameters to %s\n', filename);

% Initialize storage for posterior predictions
R_real_samples = zeros(length(freq), num_posterior_samples);
R_imag_samples = zeros(length(freq), num_posterior_samples);
alpha_posterior_samples = zeros(length(freq), num_posterior_samples);

% Compute predicted reflection and absorption coefficients for each sample
for i = 1:num_posterior_samples
    params_i = sampled_params(i, :);
    [R_sample, ~, alpha_sample, ~, ~, ~, ~] = jcal_reflection(h, phi, params_i(1), params_i(2), params_i(3), params_i(4), params_i(5), freq, airProperties);
    
    % Store real and imaginary parts separately
    R_real_samples(:, i) = real(R_sample);
    R_imag_samples(:, i) = imag(R_sample);
    alpha_posterior_samples(:, i) = alpha_sample;  % Absorption coefficient is already real
end

% Compute mean and 95% credible intervals for reflection coefficient (Real & Imaginary)
R_real_lower = prctile(R_real_samples, 2.5, 2);
R_real_upper = prctile(R_real_samples, 97.5, 2);

R_imag_lower = prctile(R_imag_samples, 2.5, 2);
R_imag_upper = prctile(R_imag_samples, 97.5, 2);

% Compute mean and 95% CI for absorption coefficient
alpha_lower = prctile(alpha_posterior_samples, 2.5, 2);
alpha_upper = prctile(alpha_posterior_samples, 97.5, 2);

% --- Posterior Predictive Check: 95% Confidence Interval Plot ---
figure;
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Real Part of Reflection Coefficient with 95% CI
nexttile;
scatter(freq, real(R_exp), 'red', '.', 'DisplayName', 'Measured (Real)');
hold on;
plot(freq, real(R_predicted), 'b-', 'LineWidth', 2, 'DisplayName', 'Posterior Mean (Real)');
fill([freq; flipud(freq)], [R_real_lower; flipud(R_real_upper)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

xlabel('Frequency (Hz)');
ylabel('Real Part of R');
legend('Experimental','Posterior mean','95 % Confidence interval');
grid on;
title('95% CI: Real Reflection Coefficient');

% Imaginary Part of Reflection Coefficient with 95% CI
nexttile;
scatter(freq, imag(R_exp), 'red', '.', 'DisplayName', 'Measured (Imaginary)');
hold on;
plot(freq, imag(R_predicted), 'b-', 'LineWidth', 2, 'DisplayName', 'Posterior Mean (Imaginary)');
fill([freq; flipud(freq)], [R_imag_lower; flipud(R_imag_upper)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

xlabel('Frequency (Hz)');
ylabel('Imaginary Part of R');
legend('Experimental','Posterior mean','95 % Confidence interval', 'Location', 'northwest');
grid on;
title('95% CI: Imaginary Reflection Coefficient');

% Sound Absorption Coefficient with 95% CI
nexttile([1 2]); % Span across two tiles
scatter(freq, alpha_exp, 'red', '.', 'DisplayName', 'Measured');
hold on;
plot(freq, alpha_predicted, 'b-', 'LineWidth', 2, 'DisplayName', 'Posterior Mean');
fill([freq; flipud(freq)], [alpha_lower; flipud(alpha_upper)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

xlabel('Frequency (Hz)');
ylabel('Sound Absorption Coefficient');
legend('Experimental','Posterior mean','95 % Confidence interval', 'Location', 'northwest');
grid on;
title('95% CI: Sound Absorption Coefficient');

% Style Modifications
set(gcf, 'Position', get(0, 'Screensize'));  % Maximize figure window
hAx = findobj(gcf, 'type', 'axes');
for i = 1:numel(hAx)
    set(hAx(i), 'LineWidth', 1, 'FontSize', 30);
    grid(hAx(i), 'on');
    box(hAx(i), 'on');
    set(hAx(i), 'XScale', 'log'); % Log scale for frequency
end

% -------------------------
% --- Random Parameter Samples Plot ---
% -------------------------
figure;
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Number of samples to plot
num_samples_to_plot = 200;
random_subset_indices = randperm(num_posterior_samples, num_samples_to_plot);

% Real Part of Reflection Coefficient for Random Samples
nexttile;
scatter(freq, real(R_exp), 'red', '.', 'DisplayName', 'Measured (Real)');
hold on;
plot(freq, real(R_predicted), 'b-', 'LineWidth', 2, 'DisplayName', 'JCAL Predicted (Imaginary)');
for i = random_subset_indices
    plot(freq, R_real_samples(:, i), 'Color', [0.6 0.6 0.6 0.3]); % Transparent gray lines
end
xlabel('Frequency (Hz)');
ylabel('Real Part of R');
legend('Experimental','Posterior mean');
grid on;
title('Random Samples: Real Reflection Coefficient');

% Imaginary Part of Reflection Coefficient for Random Samples
nexttile;
scatter(freq, imag(R_exp), 'red', '.', 'DisplayName', 'Measured (Imaginary)');
hold on;
plot(freq, imag(R_predicted), 'b-', 'LineWidth', 2, 'DisplayName', 'JCAL Predicted (Imaginary)');
for i = random_subset_indices
    plot(freq, R_imag_samples(:, i), 'Color', [0.6 0.6 0.6 0.3]); % Transparent gray lines
end
xlabel('Frequency (Hz)');
ylabel('Imaginary Part of R');
legend('Experimental','Posterior mean', 'Location', 'northwest');
grid on;
title('Random Samples: Imaginary Reflection Coefficient');

% Sound Absorption Coefficient for Random Samples
nexttile([1 2]); % Span across two tiles
scatter(freq, alpha_exp, 'red', '.', 'DisplayName', 'Measured');
hold on;
plot(freq, alpha_predicted, 'b-', 'LineWidth', 2, 'DisplayName', 'JCAL Predicted');
for i = random_subset_indices
    plot(freq, alpha_posterior_samples(:, i), 'Color', [0.6 0.6 0.6 0.3]); % Transparent gray lines
end
xlabel('Frequency (Hz)');
ylabel('Sound Absorption Coefficient');
legend('Experimental','Posterior mean', 'Location', 'northwest');
grid on;
title('Random Samples: Sound Absorption Coefficient');

% Style Modifications
set(gcf, 'Position', get(0, 'Screensize'));  % Maximize figure window
hAx = findobj(gcf, 'type', 'axes');
for i = 1:numel(hAx)
    set(hAx(i), 'LineWidth', 1, 'FontSize', 30);
    grid(hAx(i), 'on');
    box(hAx(i), 'on');
    set(hAx(i), 'XScale', 'log'); % Log scale for frequency
end


%% Save data
% Define the filename based on fiberfolder and porosityfolder
filename = sprintf('%s_%s_params.txt', fiberfolder, porosityfolder);

% Open the file for writing
fileID = fopen(filename, 'w');

% Check if file opened successfully
if fileID == -1
    error('Could not open file for writing.');
end

% Write header
fprintf(fileID, 'Inferred Parameters with Standard Deviation (SD) and Bounds\n');
fprintf(fileID, 'Fiber Type: %s\n', fiberfolder);
fprintf(fileID, 'Porosity: %s\n\n', porosityfolder);

% Print inferred parameter values and standard deviations
fprintf(fileID, 'Inferred Parameter Values (Posterior Means) and SD:\n');
fprintf(fileID, '%-30s %-15s %-15s\n', 'Parameter Name', 'Value', 'SD'); % Column headers
for i = 1:length(param_names)
    fprintf(fileID, '%-30s %.6e %.6e\n', param_names{i}, estimated_params(i), param_std(i));
end
fprintf(fileID, '\n');  % Add a newline for separation

% Print lower and upper bounds in separate columns
fprintf(fileID, 'Parameter Constraints (Lower Bound, Upper Bound):\n');
fprintf(fileID, '%-30s %-15s %-15s\n', 'Parameter Name', 'Lower Bound', 'Upper Bound'); % Column headers
for i = 1:length(param_names)
    fprintf(fileID, '%-30s %.6e %.6e\n', param_names{i}, lb(i), ub(i));
end

% Close the file
fclose(fileID);

% Confirm saving
fprintf('Saved parameters to %s\n', filename);