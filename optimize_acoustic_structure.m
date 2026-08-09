function objective_value = optimize_acoustic_structure(params, configuration, fibertype, freq, airProperties, coefficients, acrylic_first_perfect_absorption_freq_target)
    % Extract parameters for the configuration
    if configuration == 1  % Fiber + Air
        fiber_thickness = params(1);  % mm
        fiber_porosity = params(2);   % Porosity
        air_thickness = params(3);    % mm

        % Call the multi_layer_acoustics function
        [~, ~, alpha, Re, Im] = multi_layer_acoustics(configuration, fibertype, [fiber_thickness, fiber_porosity, air_thickness], freq, airProperties, coefficients);
        
    elseif configuration == 2  % Fiber + Fiber
        fiber1_thickness = params(1); % mm
        fiber1_porosity = params(2);  % Porosity
        fiber2_thickness = params(3); % mm
        fiber2_porosity = params(4);  % Porosity

        % Call the multi_layer_acoustics function
        [~, ~, alpha, Re, Im] = multi_layer_acoustics(configuration, fibertype, [fiber1_thickness, fiber1_porosity, fiber2_thickness, fiber2_porosity], freq, airProperties, coefficients);
        
    elseif configuration == 3  % MPP + Fiber
        fiber_thickness = params(1);  % mm
        fiber_porosity = params(2);   % Porosity
        mpp_perforation_rate = params(3); % Perforation rate
        mpp_micropore_r = params(4); % Micropore diameter (m)
        mpp_thickness = params(5);  % MPP thickness (mm)

        % Call the multi_layer_acoustics function
        [~, ~, alpha, Re, Im] = multi_layer_acoustics(configuration, fibertype, [fiber_thickness, fiber_porosity, mpp_perforation_rate, mpp_micropore_r, mpp_thickness], freq, airProperties, coefficients);
    else
        error('Invalid configuration');
    end
    
    % 1. Find the first frequency where absorption > 0.99 (perfect absorption)
    first_perfect_absorption_freq = NaN;
    for i = 1:length(freq)
        if alpha(i) > 0.99
            first_perfect_absorption_freq = freq(i);
            break;
        end
    end

    % Penalty for difference between target frequency and the actual first perfect absorption frequency
    freq_penalty = abs(first_perfect_absorption_freq - acrylic_first_perfect_absorption_freq_target);
    
    % 2. After the first perfect absorption, minimize deviation of Re and Im of normalized surface impedance
    % Find the index of first perfect absorption frequency
    first_abs_freq_idx = find(freq == first_perfect_absorption_freq);
    
    % Compute the difference between Re and Im of the normalized surface impedance to their ideal values
    impedance_penalty = sum(abs(Re(first_abs_freq_idx:end) - 1)) + sum(abs(Im(first_abs_freq_idx:end)));

    % Total objective value: sum of penalties (we aim to minimize these)
    objective_value = freq_penalty + impedance_penalty;
end
