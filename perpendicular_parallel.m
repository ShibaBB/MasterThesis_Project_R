%% Optimization of perpendicular and parallel assemblies

% Define frequency, porosity, and thickness ranges
freq_num = 1:1:5000;      % Frequency range (1 Hz to 5000 Hz)
phi_num = 0.92:0.001:0.99; % Porosity range (0.92 to 0.99)
h_num = (10:1:100)*1e-3;     % Thickness range (10 mm to 100 mm)

% Create mesh grids for frequency, porosity, and thickness
[freq_grid, phi_grid, h_grid] = meshgrid(freq_num, phi_num, h_num);

% Initialize result matrices for sound absorption (alpha)
acrylic_alpha_num = zeros(numel(freq_num), numel(phi_num), numel(h_num));  % 3D matrix to store sound absorption
acrylic_Zs_norm_num = zeros(numel(freq_num), numel(phi_num), numel(h_num)); % 3D matrix to store reflection coefficient

% Call the function for each combination of parameters (assuming you already have this part)
for i = 1:numel(freq_num)
    for j = 1:numel(phi_num)
        for k = 1:numel(h_num)
            [~, acrylic_Zs_norm, acrylic_alpha_num(i,j,k), ~, ~, ~, ~] = ...
                jcal_s(1, h_num(k), phi_num(j), coefficients, freq_num(i), airProperties);
        end
    end
end

% Initialize result matrices for sound absorption (alpha)
wool_alpha_num = zeros(numel(freq_num), numel(phi_num), numel(h_num));  % 3D matrix to store sound absorption
wool_Zs_norm_num = zeros(numel(freq_num), numel(phi_num), numel(h_num)); % 3D matrix to store reflection coefficient

% Call the function for each combination of parameters (assuming you already have this part)
for i = 1:numel(freq_num)
    for j = 1:numel(phi_num)
        for k = 1:numel(h_num)
            [~, wool_Zs_norm, wool_alpha_num(i,j,k), ~, ~, ~, ~] = ...
                jcal_s(3, h_num(k), phi_num(j), coefficients, freq_num(i), airProperties);
        end
    end
end

%% Figures for fixed porosity or thickness, acrylic fiber
% Fixed Porosity (phi = 0.96) - plot sound absorption vs frequency and thickness
phi_fixed = 0.96;
acrylic_phi_index = find(abs(phi_num - phi_fixed) < 1e-6, 1);  
if isempty(acrylic_phi_index)
    error('No matching phi value found for acrylic fiber!');
end

acrylic_alpha_fixed_phi = squeeze(acrylic_alpha_num(:, acrylic_phi_index, :));  % Extract the data for fixed phi

% Create the figure
figure;
contourf(freq_num, h_num*1000, acrylic_alpha_fixed_phi', 'LineStyle', 'none', ...
    'LevelList', linspace(min(acrylic_alpha_fixed_phi(:)), max(acrylic_alpha_fixed_phi(:)), 50)); % Plot for fixed phi
colormap(parula);
colorbar;
xlabel('Frequency (Hz)', 'FontSize', 18);
ylabel('Thickness (mm)', 'FontSize', 18);
title(['Sound absorption for acrylic with \phi = ', num2str(phi_fixed*100), '%'], 'FontSize', 14);
grid on;
hold on;
axis([1 5000 10 100]);
set(gca, 'LineWidth', 1, 'FontSize', 16);

% Loop through all thickness values and find the first perfect absorption
for row_index = 1:size(acrylic_alpha_fixed_phi, 2)  % Loop through thickness index
    % Ensure the row_index corresponds correctly to h_num
    if row_index <= length(h_num)
        % Find the FIRST column index where alpha_fixed_phi exceeds 0.99
        first_col_index = find(acrylic_alpha_fixed_phi(:, row_index) > 0.99, 1, 'first');
        
        % If a valid index is found, plot it
        if ~isempty(first_col_index)
            scatter(freq_num(first_col_index), h_num(row_index) * 1000, ...
                50, 'red', 'filled', 'LineWidth', 1.5);
        end
    end
end

% Fixed Thickness (h = 30 mm) - plot sound absorption vs frequency and porosity
h_fixed = 30;  % Fixed thickness (mm)
[~, acrylic_h_index] = min(abs(h_num - h_fixed*1e-3));  % Find the index for 30mm thickness
acrylic_alpha_fixed_h = squeeze(acrylic_alpha_num(:, :, acrylic_h_index));  % Extract the data for fixed thickness

% Create the figure
figure;
contourf(freq_num, phi_num, acrylic_alpha_fixed_h', 'LineStyle', 'none', ...
    'LevelList', linspace(min(acrylic_alpha_fixed_h(:)), max(acrylic_alpha_fixed_h(:)), 50));
colormap(parula);
colorbar;
xlabel('Frequency (Hz)', 'FontSize', 18);
ylabel('Porosity (\phi)', 'FontSize', 18);
title(['Sound absorption for acrylic with h = ', num2str(h_fixed), ' mm'], 'FontSize', 14);
grid on;
hold on;
axis([1 5000 0.92 0.99]);
set(gca, 'LineWidth', 1, 'FontSize', 16);

% Loop through all porosity values and find the first perfect absorption
for phi_index = 1:size(acrylic_alpha_fixed_h, 2)  % Loop through porosity index
    if phi_index <= length(phi_num)
        % Find first occurrence where absorption > 0.998
        first_col_index = find(acrylic_alpha_fixed_h(:, phi_index) > 0.99, 1, 'first');
        
        % Plot if found
        if ~isempty(first_col_index)
            scatter(freq_num(first_col_index), phi_num(phi_index), ...
                50, 'red', 'filled', 'LineWidth', 1.5);
        end
    end
end

%% Figures for fixed porosity or thickness, wool fiber
% Fixed Porosity (phi = 0.96) - plot sound absorption vs frequency and thickness
wool_phi_index = find(abs(phi_num - phi_fixed) < 1e-6, 1);
if isempty(wool_phi_index)
    error('No matching phi value found for wool fiber!');
end

wool_alpha_fixed_phi = squeeze(wool_alpha_num(:, wool_phi_index, :));  % Extract the data for fixed phi

% Create the figure
figure;
contourf(freq_num, h_num*1000, wool_alpha_fixed_phi', 'LineStyle', 'none', ...
    'LevelList', linspace(min(wool_alpha_fixed_phi(:)), max(wool_alpha_fixed_phi(:)), 50)); % Plot for fixed phi
colormap(parula);
colorbar;
xlabel('Frequency (Hz)', 'FontSize', 18);
ylabel('Thickness (mm)', 'FontSize', 18);
title(['Sound absorption for wool with \phi = ', num2str(phi_fixed*100), '%'], 'FontSize', 14);
grid on;
hold on;
axis([1 5000 10 100]);
set(gca, 'LineWidth', 1, 'FontSize', 16);

% Loop through all thickness values and find the first perfect absorption
for row_index = 1:size(wool_alpha_fixed_phi, 2)  % Loop through thickness index
    % Ensure the row_index corresponds correctly to h_num
    if row_index <= length(h_num)
        % Find the FIRST column index where alpha_fixed_phi exceeds 0.99
        first_col_index = find(wool_alpha_fixed_phi(:, row_index) > 0.99, 1, 'first');
        
        % If a valid index is found, plot it
        if ~isempty(first_col_index)
            scatter(freq_num(first_col_index), h_num(row_index) * 1000, ...
                50, 'red', 'filled', 'LineWidth', 1.5);
        end
    end
end

% Fixed Thickness (h = 30 mm) - plot sound absorption vs frequency and porosity
[~, wool_h_index] = min(abs(h_num - h_fixed*1e-3));  % Find the index for 30mm thickness
wool_alpha_fixed_h = squeeze(wool_alpha_num(:, :, wool_h_index));  % Extract the data for fixed thickness

% Create the figure
figure;
contourf(freq_num, phi_num, wool_alpha_fixed_h', 'LineStyle', 'none', ...
    'LevelList', linspace(min(wool_alpha_fixed_h(:)), max(wool_alpha_fixed_h(:)), 50));
colormap(parula);
colorbar;
xlabel('Frequency (Hz)', 'FontSize', 18);
ylabel('Porosity (\phi)', 'FontSize', 18);
title(['Sound absorption for wool with h = ', num2str(h_fixed), ' mm'], 'FontSize', 14);
grid on;
hold on;
axis([1 5000 0.92 0.99]);
set(gca, 'LineWidth', 1, 'FontSize', 16);

% Loop through all porosity values and find the first perfect absorption
for phi_index = 1:size(wool_alpha_fixed_h, 2)  % Loop through porosity index
    if phi_index <= length(phi_num)
        % Find first occurrence where absorption > 0.99
        first_col_index = find(wool_alpha_fixed_h(:, phi_index) > 0.99, 1, 'first');
        
        % Plot if found
        if ~isempty(first_col_index)
            scatter(freq_num(first_col_index), phi_num(phi_index), ...
                50, 'red', 'filled', 'LineWidth', 1.5);
        end
    end
end

%% Optimization，multi-objective optimization
% Perpendicular assembly

% Define target thickness values (in mm) for acrylic material
acrylic_h_values = [40, 80];  % Thickness values (mm) for acrylic material

% Initialize variables for storing the first perfect absorption frequencies and corresponding porosities
acrylic_first_perfect_absorption_freq = NaN(size(acrylic_h_values));  % Initialize with NaN for each thickness
acrylic_corresponding_porosity = NaN(size(acrylic_h_values));  % Initialize corresponding porosities

% Create a figure for plotting the sound absorption curves
figure;
hold on;

% Loop through each thickness value for acrylic material
for acrylic_idx = 1:length(acrylic_h_values)
    acrylic_h_fixed = acrylic_h_values(acrylic_idx);  % Set thickness for acrylic material
    [~, acrylic_h_index] = min(abs(h_num - acrylic_h_fixed * 1e-3));  % Find index for thickness in the data
    acrylic_alpha_fixed_h = squeeze(acrylic_alpha_num(:, :, acrylic_h_index));  % Extract data for fixed thickness
    
    % Initialize variable to store the lowest frequency where absorption > 0.99 for the current thickness
    acrylic_first_perfect_absorption_freq_current = NaN;
    acrylic_corresponding_porosity_current = NaN;
    
    % Loop through porosity index to find the first perfect absorption frequency
    for acrylic_phi_idx = 1:size(acrylic_alpha_fixed_h, 2)  % Loop through porosity index
        if acrylic_phi_idx <= length(phi_num)
            % Find first occurrence where absorption > 0.99 for each porosity
            acrylic_first_col_index = find(acrylic_alpha_fixed_h(:, acrylic_phi_idx) > 0.99, 1, 'first');
            
            % If a valid index is found, get the corresponding frequency and porosity
            if ~isempty(acrylic_first_col_index)
                % If this is the first valid frequency or lower than the previous one, update
                if isnan(acrylic_first_perfect_absorption_freq_current) || freq_num(acrylic_first_col_index) < acrylic_first_perfect_absorption_freq_current
                    acrylic_first_perfect_absorption_freq_current = freq_num(acrylic_first_col_index);  % Update the lowest frequency
                    acrylic_corresponding_porosity_current = phi_num(acrylic_phi_idx);  % Update corresponding porosity
                end
            end
        end
    end
    
    % Store the result for the current thickness in the context of acrylic material
    acrylic_first_perfect_absorption_freq(acrylic_idx) = acrylic_first_perfect_absorption_freq_current;
    acrylic_corresponding_porosity(acrylic_idx) = acrylic_corresponding_porosity_current;
    
    % Plot the absorption curve for the current thickness at the frequency of perfect absorption
    % Plot only the curve where perfect absorption occurs at the lowest frequency
    if ~isnan(acrylic_first_perfect_absorption_freq_current)
        % Plot the sound absorption curve for the corresponding porosity
        plot(freq_num, squeeze(acrylic_alpha_fixed_h(:, acrylic_corresponding_porosity_current == phi_num)), ...
        'LineWidth', 3, 'DisplayName', ...
        [num2str(acrylic_h_fixed) ' mm, \phi = ' num2str(acrylic_corresponding_porosity_current) ...
        ', f_{lowest} = ' num2str(acrylic_first_perfect_absorption_freq_current) ' Hz']);
    end
end

% Customize the plot
xlabel('Frequency (Hz)', 'FontSize', 18);
ylabel('Sound Absorption Coefficient', 'FontSize', 18);
legend('show', 'Location', 'southeast', 'box', 'off', 'FontSize', 16);
title({'Sound Absorption of Acrylic Fiber:', ...
       'Perfect Absorption at Lowest Frequency'}, 'FontSize', 12);

% Apply formatting to axes
set(gca, 'LineWidth', 2, 'FontSize', 16);
grid on;
box on;

% Display the results for each thickness
disp('First perfect absorption frequency for each acrylic thickness:');
disp(table(acrylic_h_values(:), acrylic_first_perfect_absorption_freq(:), acrylic_corresponding_porosity(:), ...
    'VariableNames', {'Thickness_mm', 'First_Perfect_Absorption_Freq_Hz', 'Corresponding_Porosity'}));

%% Optimization of 80 mm thickness structure for different configurations
% Define the configurations 
configurations = {'Fiber_Air', 'Fiber_Fiber', 'MPP_Fiber'};
fibertype = 1; % 1: Acrylic fiber; 2: Silk fiber; 3. Wool fiber.

optimized_results = struct(); % Store results

% Set GA options
options = optimoptions('ga', ...
    'PopulationSize', 20, ...
    'MaxGenerations', 50, ...
    'MaxStallGenerations', 5, ...
    'ConstraintTolerance', 1e-6, ...  % Lower tolerance for stricter enforcement
    'Display', 'iter', ...
    'TolFun', 1e-6, ...  % Reduce function tolerance
    'TolCon', 1e-6, ...  % Reduce constraint tolerance
    'MutationFcn', @mutationadaptfeasible, ... % Use constraint-adaptive mutation
    'CrossoverFcn', @crossoversinglepoint, ... % Use single-point crossover for reproducibility
    'EliteCount', 2);  % Keep the best solutions through generations

% Initialize parameter storage for GA and PSO
params1 = [];
params2 = [];
params3 = [];

% Loop through configurations and optimize using both GA and PSO
for config_idx = 1:length(configurations)
    configuration = config_idx;
    
    % Set variable bounds
    switch configuration
        case 1  % Fiber + Air
            num_vars = 3;
            lower_bounds = [10e-3, 0.92, 10e-3];
            upper_bounds = [70e-3, 0.99, 70e-3];

        case 2  % Fiber + Fiber
            num_vars = 4;
            lower_bounds = [10e-3, 0.92, 10e-3, 0.92];
            upper_bounds = [70e-3, 0.99, 70e-3, 0.99];

        case 3  % MPP + Fiber
            num_vars = 5;
            lower_bounds = [10e-3, 0.92, 0.01, 0.1e-3, 0.1e-3];
            upper_bounds = [70e-3, 0.99, 0.05, 5e-3, 5e-3];
    end
    
    % GA Optimization
    [optimal_params_ga, ~] = ga(@(params) optimize_acoustic_structure(params, configuration, fibertype, freq_num, airProperties, coefficients, acrylic_first_perfect_absorption_freq(2)), ...
                                num_vars, [], [], [], [], lower_bounds, upper_bounds, ...
                                @(params) thickness_constraint(params, configuration), ga_options);

    % PSO Optimization
    [optimal_params_pso, ~] = particleswarm(@(params) optimize_acoustic_structure(params, configuration, fibertype, freq_num, airProperties, coefficients, acrylic_first_perfect_absorption_freq(2)), ...
                                            num_vars, lower_bounds, upper_bounds, pso_options);

    % Store parameters for both GA and PSO
    if config_idx == 1
        params1 = optimal_params_ga;
        params1_pso = optimal_params_pso;
    elseif config_idx == 2
        params2 = optimal_params_ga;
        params2_pso = optimal_params_pso;
    elseif config_idx == 3
        params3 = optimal_params_ga;
        params3_pso = optimal_params_pso;
    end
end

% Initialize a figure for the plot
figure;
hold on;

% Plot single-layer (80 mm) absorption curve
plot(freq_num, acrylic_alpha_80mm, 'k-', 'LineWidth', 3, 'DisplayName', ...
     ['80 mm, \phi = ' num2str(acrylic_corresponding_porosity(acrylic_h_80mm_idx)*1e+2, '%.2f') '%, '...
     ', f_{lowest} = ' num2str(acrylic_first_perfect_absorption_freq(acrylic_h_80mm_idx), '%.2f') ' Hz']);

% Plot for Fiber + Air (Configuration 1)
[Reflect1, Zs_norm1, alpha1, Re1, Im1] = multi_layer_acoustics(1, fibertype, params1, freq_num, airProperties, coefficients);
total_thickness1 = (params1(1) + params1(3)) * 1e+3; % Convert to mm
reduction1 = (80 - total_thickness1) / 80 * 100; % Compute thickness reduction percentage

display_name_1 = ['Fiber + Air (' num2str(reduction1, '%.2f') '% thickness reduction)' newline ...
                  num2str(params1(1)*1e+3, '%.2f') ' mm, ' num2str(params1(2)*100, '%.2f') '% + ' ...
                  num2str(params1(3)*1e+3, '%.2f') ' mm'];
plot(freq_num, alpha1, 'LineWidth', 3, 'DisplayName', display_name_1);

% Plot for Fiber + Fiber (Configuration 2)
[Reflect2, Zs_norm2, alpha2, Re2, Im2] = multi_layer_acoustics(2, fibertype, params2, freq_num, airProperties, coefficients);
total_thickness2 = (params2(1) + params2(3)) * 1e+3; % Convert to mm
reduction2 = (80 - total_thickness2) / 80 * 100; % Compute thickness reduction percentage
display_name_2 = ['Fiber + Fiber (' num2str(reduction2, '%.2f') '% thickness reduction)' newline ...
                  num2str(params2(1)*1e+3, '%.2f') ' mm, ' num2str(params2(2)*100, '%.2f') '% + ' ...
                  num2str(params2(3)*1e+3, '%.2f') ' mm, ' num2str(params2(4)*100, '%.2f') '%'];
plot(freq_num, alpha2, 'LineWidth', 3, 'DisplayName', display_name_2);

% Plot for MPP + Fiber (Configuration 3)
[Reflect3, Zs_norm3, alpha3, Re3, Im3] = multi_layer_acoustics(3, fibertype, params3, freq_num, airProperties, coefficients);
total_thickness3 = (params3(1) + params3(5)) * 1e+3; % Convert to mm
reduction3 = (80 - total_thickness3) / 80 * 100; % Compute thickness reduction percentage

display_name_3 = ['MPP + Fiber (' num2str(reduction3, '%.2f') '% thickness reduction)' newline ...
                  num2str(params3(3)*100, '%.2f') '%, ' num2str(params3(4)*1e+3, '%.2f') ' mm, ' num2str(params3(5)*1e+3, '%.2f') ' mm +' ...
                  ' ' num2str(params3(1)*1e+3, '%.2f') ' mm, ' num2str(params3(2)*100, '%.2f') '%'];
plot(freq_num, alpha3, 'LineWidth', 3, 'DisplayName', display_name_3);

% Customize the plot
xlabel('Frequency (Hz)', 'FontSize', 18);
ylabel('Sound Absorption Coefficient', 'FontSize', 18);
legend('show', 'Location', 'south', 'box', 'off', 'FontSize', 16);
title({'Comparison of Optimized Pependicular Assembly Absorption', ...
       'with Single-layer (80mm)'}, 'FontSize', 12);

% Apply formatting to axes
set(gca, 'LineWidth', 2, 'FontSize', 16);
grid on;
box on;

% Initialize a figure for the plot, PSO
figure;
hold on;

% Plot single-layer (80 mm) absorption curve
plot(freq_num, acrylic_alpha_80mm, 'k-', 'LineWidth', 3, 'DisplayName', ...
     ['80 mm, \phi = ' num2str(acrylic_corresponding_porosity(acrylic_h_80mm_idx)*1e+2, '%.2f') '%, '...
     ' f_{lowest} = ' num2str(acrylic_first_perfect_absorption_freq(acrylic_h_80mm_idx), '%.2f') ' Hz']);

% Plot for Fiber + Air (Configuration 1)
[Reflect4, Zs_norm4, alpha4, Re4, Im4] = multi_layer_acoustics(1, fibertype, params1_pso, freq_num, airProperties, coefficients);
total_thickness1 = (params1(1) + params1(3)) * 1e+3; % Convert to mm
reduction1 = (80 - total_thickness1) / 80 * 100; % Compute thickness reduction percentage

display_name_4 = ['Fiber + Air (' num2str(reduction1, '%.2f') '% thickness reduction)' newline ...
                  num2str(params1(1)*1e+3, '%.2f') ' mm, ' num2str(params1(2)*100, '%.2f') '% + ' ...
                  num2str(params1(3)*1e+3, '%.2f') ' mm'];
plot(freq_num, alpha4, '--', 'LineWidth', 3, 'DisplayName', display_name_4);

% Plot for Fiber + Fiber (Configuration 2)
[Reflect5, Zs_norm5, alpha5, Re5, Im5] = multi_layer_acoustics(2, fibertype, params2_pso, freq_num, airProperties, coefficients);
total_thickness2 = (params2(1) + params2(3)) * 1e+3; % Convert to mm
reduction2 = (80 - total_thickness2) / 80 * 100; % Compute thickness reduction percentage
display_name_5 = ['Fiber + Fiber (' num2str(reduction2, '%.2f') '% thickness reduction)' newline ...
                  num2str(params2(1)*1e+3, '%.2f') ' mm, ' num2str(params2(2)*100, '%.2f') '% + ' ...
                  num2str(params2(3)*1e+3, '%.2f') ' mm, ' num2str(params2(4)*100, '%.2f') '%'];
plot(freq_num, alpha5, '--', 'LineWidth', 3, 'DisplayName', display_name_5);

% Plot for MPP + Fiber (Configuration 3)
[Reflect6, Zs_norm6, alpha6, Re6, Im6] = multi_layer_acoustics(3, fibertype, params3_pso, freq_num, airProperties, coefficients);
total_thickness3 = (params3(1) + params3(5)) * 1e+3; % Convert to mm
reduction3 = (80 - total_thickness3) / 80 * 100; % Compute thickness reduction percentage

display_name_6 = ['MPP + Fiber (' num2str(reduction3, '%.2f') '% thickness reduction)' newline ...
                  num2str(params3(3)*100, '%.2f') '%, ' num2str(params3(4)*1e+3, '%.2f') ' mm, ' num2str(params3(5)*1e+3, '%.2f') ' mm +' ...
                  ' ' num2str(params3(1)*1e+3, '%.2f') ' mm, ' num2str(params3(2)*100, '%.2f') '%'];
plot(freq_num, alpha6, '--', 'LineWidth', 3, 'DisplayName', display_name_6);

% Customize the plot
xlabel('Frequency (Hz)', 'FontSize', 18);
ylabel('Sound Absorption Coefficient', 'FontSize', 18);
legend('show', 'Location', 'eastoutside', 'box', 'off', 'FontSize', 22);
title({'Comparison of Optimized Pependicular Assembly Absorption', ...
       'with Single-layer (80mm)'}, 'FontSize', 12);

% Apply formatting to axes
set(gca, 'LineWidth', 2, 'FontSize', 16);
grid on;
box on;


%% Parallel Structure Optimization 

% Define configurations
configurations = {'Fiber_Air', 'Fiber_Fiber', 'MPP_Fiber'};
fiber_type_parallel = 1; % Acrylic fiber

% Set area ratios (50:50 for two parallel assemblies)
area_ratios_parallel = [0.5, 0.5];

% Define target first perfect absorption frequency
if ~exist('acrylic_first_perfect_absorption_freq', 'var')
    error('Error: acrylic_first_perfect_absorption_freq is missing!');
end
target_freq_parallel = acrylic_first_perfect_absorption_freq(2);

% Storage for optimized results
optimized_results_parallel = struct();

% GA Optimization Options
options_parallel = optimoptions('ga', ...
    'PopulationSize', 20, ...
    'MaxGenerations', 50, ...
    'MaxStallGenerations', 5, ...
    'ConstraintTolerance', 1e-6, ...
    'Display', 'iter', ...
    'MutationFcn', @mutationadaptfeasible, ...
    'CrossoverFcn', @crossoversinglepoint, ...
    'EliteCount', 2);

% PSO Optimization Options
options_pso_parallel = optimoptions('particleswarm', ...
    'SwarmSize', 30, ...
    'MaxIterations', 50, ...
    'Display', 'iter', ...
    'TolFun', 1e-6);

% Initialize parameter storage for GA and PSO
params_parallel_ga = cell(1, length(configurations));
params_parallel_pso = cell(1, length(configurations));

% Loop through each configuration for parallel structure
for config_idx = 1:length(configurations)
    configuration_parallel = config_idx;
    
    % Define optimization variable count and bounds
    switch configuration_parallel
        case 1  % Fiber + Air
            num_vars = 3;
            lower_bounds_parallel = [10e-3, 0.92, 10e-3];
            upper_bounds_parallel = [80e-3, 0.99, 80e-3];

        case 2  % Fiber + Fiber
            num_vars = 4;
            lower_bounds_parallel = [10e-3, 0.92, 10e-3, 0.92];
            upper_bounds_parallel = [80e-3, 0.99, 80e-3, 0.99];

        case 3  % MPP + Fiber (with air cavity)
            num_vars = 6;
            lower_bounds_parallel = [10e-3, 0.92, 0.01, 0.1e-3, 0.1e-3, 10e-3];
            upper_bounds_parallel = [80e-3, 0.99, 0.05, 5e-3, 5e-3, 80e-3];
    end
    
    % Ensure lower bounds are not too close to zero
    lower_bounds_parallel = lower_bounds_parallel + 1e-6;
    
    % Run GA optimization with error handling
    try
        [optimal_params_parallel_ga, fval_parallel_ga] = ga(@(params) optimize_parallel_structure(params, configuration_parallel, fiber_type_parallel, freq, airProperties, coefficients, area_ratios_parallel, target_freq_parallel), ...
                                num_vars, [], [], [], [], lower_bounds_parallel, upper_bounds_parallel, ...
                                @(params) thickness_constraint_parallel(params, configuration_parallel), options_parallel);
    catch ME
        warning('GA optimization failed for %s: %s', configurations{config_idx}, ME.message);
        continue;
    end
    
    % Run PSO optimization with error handling
    try
        [optimal_params_parallel_pso, fval_parallel_pso] = particleswarm(@(params) optimize_parallel_structure(params, configuration_parallel, fiber_type_parallel, freq, airProperties, coefficients, area_ratios_parallel, target_freq_parallel), ...
                                            num_vars, lower_bounds_parallel, upper_bounds_parallel, options_pso_parallel);
    catch ME
        warning('PSO optimization failed for %s: %s', configurations{config_idx}, ME.message);
        continue;
    end
    
    % Store optimized parameters and objective function values
    optimized_results_parallel.(configurations{config_idx}).GA.params = optimal_params_parallel_ga;
    optimized_results_parallel.(configurations{config_idx}).GA.objective_value = fval_parallel_ga;
    optimized_results_parallel.(configurations{config_idx}).PSO.params = optimal_params_parallel_pso;
    optimized_results_parallel.(configurations{config_idx}).PSO.objective_value = fval_parallel_pso;

    params_parallel_ga{config_idx} = optimal_params_parallel_ga;
    params_parallel_pso{config_idx} = optimal_params_parallel_pso;
    
    % Compute sound absorption for optimized structure using ASM
    try
        [~, ~, alpha_optimized_parallel_ga, ~, ~] = parallel_acoustic_structure(configuration_parallel, fiber_type_parallel, optimal_params_parallel_ga, freq, airProperties, coefficients, area_ratios_parallel);
        [~, ~, alpha_optimized_parallel_pso, ~, ~] = parallel_acoustic_structure(configuration_parallel, fiber_type_parallel, optimal_params_parallel_pso, freq, airProperties, coefficients, area_ratios_parallel);
    catch ME
        warning('Error in acoustic calculation for %s: %s', configurations{config_idx}, ME.message);
        continue;
    end
    
    % Validate results
    if max(alpha_optimized_parallel_ga) > 1 || min(alpha_optimized_parallel_ga) < 0
        warning('Unphysical results detected for GA in %s, skipping...', configurations{config_idx});
        continue;
    end
    if max(alpha_optimized_parallel_pso) > 1 || min(alpha_optimized_parallel_pso) < 0
        warning('Unphysical results detected for PSO in %s, skipping...', configurations{config_idx});
        continue;
    end

    optimized_results_parallel.(configurations{config_idx}).GA.absorption = alpha_optimized_parallel_ga;
    optimized_results_parallel.(configurations{config_idx}).PSO.absorption = alpha_optimized_parallel_pso;
end

% Save optimized data
save('optimized_parallel_data.mat', 'optimized_results_parallel', 'params_parallel_ga', 'params_parallel_pso');

% Initialize a figure for the plot
figure;
hold on;

% Plot single-layer (80 mm) absorption curve
plot(freq_num, acrylic_alpha_80mm, 'k-', 'LineWidth', 3, 'DisplayName', ...
     ['80 mm, \phi = ' num2str(acrylic_corresponding_porosity(acrylic_h_80mm_idx) * 1e+2, '%.2f') '%, ' ...
     'f_{lowest} = ' num2str(acrylic_first_perfect_absorption_freq(acrylic_h_80mm_idx), '%.2f') ' Hz']);

% Loop through each configuration for parallel structure
for config_idx = 1:length(configurations)
    params = optimized_results_parallel.(configurations{config_idx}).GA.params;
    [~, ~, alpha, ~, ~] = parallel_acoustic_structure(config_idx, fiber_type_parallel, params, freq_num, airProperties, coefficients, area_ratios_parallel);
    
    % Compute thickness reduction
    if config_idx == 1
        max_thickness = params(1); % Fiber thickness only
    elseif config_idx == 2
        max_thickness = max(params([1,3])); % Largest fiber thickness
    elseif config_idx == 3
        max_thickness = max(params(1), sum(params([5,6]))); % Max fiber or MPP+air
    end
    thickness_reduction = (80e-3 - max_thickness) / 80e-3 * 100;
    
    % Create display name
    if config_idx == 1  % Fiber + Air
        display_name = ['Fiber (' num2str(thickness_reduction, '%.2f') '% thickness reduction)' newline ...
            num2str(params(1)*1e+3, '%.2f') ' mm, ' num2str(params(2)*100, '%.2f') '%'];
    elseif config_idx == 2  % Fiber + Fiber
        display_name = ['Fiber + Fiber (' num2str(thickness_reduction, '%.2f') '% thickness reduction)' newline ...
            num2str(params(1)*1e+3, '%.2f') ' mm, ' num2str(params(2)*100, '%.2f') '% + ' ...
            num2str(params(3)*1e+3, '%.2f') ' mm, ' num2str(params(4)*100, '%.2f') '%'];
    elseif config_idx == 3  % MPP + Fiber
        display_name = ['MPP + Fiber (' num2str(thickness_reduction, '%.2f') '% thickness reduction)' newline ...
            num2str(params(3)*100, '%.2f') '%, ' num2str(params(4)*1e+3, '%.2f') ' mm, ' num2str(params(5)*1e+3, '%.2f') ' mm, ' num2str(params(6)*1e+3, '%.2f') ' mm +' ...
            ' ' num2str(params(1)*1e+3, '%.2f') ' mm, ' num2str(params(2)*100, '%.2f') '%'];
    end
    
    % Plot absorption
    plot(freq_num, alpha, '--', 'LineWidth', 3, 'DisplayName', display_name);
end

% Finalize graph settings
xlabel('Frequency (Hz)', 'FontSize', 18);
ylabel('Sound Absorption Coefficient', 'FontSize', 18);
legend('show', 'Location', 'south', 'box', 'off', 'FontSize', 16);
title({'Comparison of Optimized Parallel Assembly Absorption', ...
       'with Single-layer (80mm)'}, 'FontSize', 12);
set(gca, 'LineWidth', 2, 'FontSize', 16);
grid on;
box on;

% Initialize a figure for the plot
figure;
hold on;

% Plot single-layer (80 mm) absorption curve
plot(freq_num, acrylic_alpha_80mm, 'k-', 'LineWidth', 3, 'DisplayName', ...
     ['80 mm, \phi = ' num2str(acrylic_corresponding_porosity(acrylic_h_80mm_idx) * 1e+2, '%.2f') '%, ' ...
     'f_{lowest} = ' num2str(acrylic_first_perfect_absorption_freq(acrylic_h_80mm_idx), '%.2f') ' Hz']);

% Loop through each configuration for parallel structure
for config_idx = 1:length(configurations)
    params = optimized_results_parallel.(configurations{config_idx}).PSO.params;
    [~, ~, alpha, ~, ~] = parallel_acoustic_structure(config_idx, fiber_type_parallel, params, freq_num, airProperties, coefficients, area_ratios_parallel);
    
    % Compute thickness reduction
    if config_idx == 1
        max_thickness = params(1); % Fiber thickness only
    elseif config_idx == 2
        max_thickness = max(params([1,3])); % Largest fiber thickness
    elseif config_idx == 3
        max_thickness = max(params(1), sum(params([5,6]))); % Max fiber or MPP+air
    end
    thickness_reduction = (80e-3 - max_thickness) / 80e-3 * 100;
    
    % Create display name
    if config_idx == 1  % Fiber + Air
        display_name = ['Fiber (' num2str(thickness_reduction, '%.2f') '% thickness reduction)' newline ...
            num2str(params(1)*1e+3, '%.2f') ' mm, ' num2str(params(2)*100, '%.2f') '%'];
    elseif config_idx == 2  % Fiber + Fiber
        display_name = ['Fiber + Fiber (' num2str(thickness_reduction, '%.2f') '% thickness reduction)' newline ...
            num2str(params(1)*1e+3, '%.2f') ' mm, ' num2str(params(2)*100, '%.2f') '% + ' ...
            num2str(params(3)*1e+3, '%.2f') ' mm, ' num2str(params(4)*100, '%.2f') '%'];
    elseif config_idx == 3  % MPP + Fiber
        display_name = ['MPP + Fiber (' num2str(thickness_reduction, '%.2f') '% thickness reduction)' newline ...
            num2str(params(3)*100, '%.2f') '%, ' num2str(params(4)*1e+3, '%.2f') ' mm, ' num2str(params(5)*1e+3, '%.2f') ' mm, ' num2str(params(6)*1e+3, '%.2f') ' mm +' ...
            ' ' num2str(params(1)*1e+3, '%.2f') ' mm, ' num2str(params(2)*100, '%.2f') '%'];
    end
    
    % Plot absorption
    plot(freq_num, alpha, '--', 'LineWidth', 3, 'DisplayName', display_name);
end

% Finalize graph settings
xlabel('Frequency (Hz)', 'FontSize', 18);
ylabel('Sound Absorption Coefficient', 'FontSize', 18);
legend('show', 'Location', 'eastoutside', 'box', 'off', 'FontSize', 22);
title({'Comparison of Optimized Parallel Assembly Absorption', ...
       'with Single-layer (80mm)'}, 'FontSize', 12);
set(gca, 'LineWidth', 2, 'FontSize', 16);
grid on;
box on;