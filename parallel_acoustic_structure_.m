function [Reflect, Zs_norm, alpha, Re, Im] = parallel_acoustic_structure(configuration, fibertype, parameters, freq, airProperties, coefficients, area_ratios)
    % Compute sound absorption, reflection coefficient, and surface impedance
    % using the Transfer Matrix Method for parallel assemblies.
    %
    % Based on: "Transfer matrix method applied to the parallel assembly of sound absorbing materials"
    %
    % Inputs:
    % - configuration: 1 (Fibers + Air), 2 (Fibers + Fibers), 3 (MPP + Fibers)
    % - fibertype: 1 (Acrylic), 2 (Silk), 3 (Wool)
    % - parameters: Optimized material properties
    % - freq: Frequency vector (Hz)
    % - airProperties: Struct containing air properties
    % - coefficients: Fiber material coefficients (for JCAL model)
    % - area_ratios: Fractional surface area occupied by each material in the parallel structure
    %
    % Outputs:
    % - Reflect: Reflection coefficient
    % - Zs_norm: Normalized surface impedance
    % - alpha: Sound absorption coefficient
    % - Re: Real part of normalized surface impedance
    % - Im: Imaginary part of normalized surface impedance
    
    % Extract air properties
    Z_air = airProperties.impedance;
    v_sound = airProperties.speed_of_sound;
    
    % Initialize result vectors
    num_freq = length(freq);
    Zs = zeros(1, num_freq);
    Reflect = zeros(1, num_freq);
    Zs_norm = zeros(1, num_freq);
    Re = zeros(1, num_freq);
    Im = zeros(1, num_freq);
    alpha = zeros(1, num_freq);
    
    % Loop over frequency range
    for i = 1:num_freq
        current_freq = freq(i);
        
        % Compute impedance and transfer matrices for each parallel element
        if configuration == 1  % Fibers + Air
            % Extract parameters
            fiber_thickness = parameters(1);
            fiber_porosity = parameters(2);
            air_thickness = parameters(3);
            
            % Compute fiber layer properties
            [~, ~, ~, ~, ~, Z_fiber, k_fiber] = jcal_s(fibertype, fiber_thickness, fiber_porosity, coefficients, current_freq, airProperties);
            
            % Compute air layer properties
            k_air = 2 * pi * current_freq / v_sound;
            Z_air_layer = Z_air * coth(1i * k_air * air_thickness);
            
            % Compute parallel admittance matrix
            Y_parallel = area_ratios(1) / Z_fiber + area_ratios(2) / Z_air_layer;
            Z_parallel = 1 / Y_parallel;
            
        elseif configuration == 2  % Fibers + Fibers
            % Extract parameters
            fiber1_thickness = parameters(1);
            fiber1_porosity = parameters(2);
            fiber2_thickness = parameters(3);
            fiber2_porosity = parameters(4);
            
            % Compute properties of each fiber layer
            [~, ~, ~, ~, ~, Z_fiber1, k_fiber1] = jcal_s(fibertype, fiber1_thickness, fiber1_porosity, coefficients, current_freq, airProperties);
            [~, ~, ~, ~, ~, Z_fiber2, k_fiber2] = jcal_s(fibertype, fiber2_thickness, fiber2_porosity, coefficients, current_freq, airProperties);
            
            % Compute parallel admittance matrix
            Y_parallel = area_ratios(1) / Z_fiber1 + area_ratios(2) / Z_fiber2;
            Z_parallel = 1 / Y_parallel;
            
        elseif configuration == 3  % MPP + Fibers
            % Extract parameters
            fiber_thickness = parameters(1);
            fiber_porosity = parameters(2);
            mpp_theta = parameters(3);  % MPP perforation rate
            mpp_r = parameters(4);  % MPP micropore diameter
            mpp_t = parameters(5);  % MPP thickness
            
            % Compute properties of fiber layer
            [~, ~, ~, ~, ~, Z_fiber, k_fiber] = jcal_s(fibertype, fiber_thickness, fiber_porosity, coefficients, current_freq, airProperties);
            
            % Compute MPP impedance
            Z_MPP = MPP1(current_freq, mpp_theta, mpp_r, mpp_t, airProperties);
            
            % Compute parallel admittance matrix
            Y_parallel = area_ratios(1) / Z_MPP + area_ratios(2) / Z_fiber;
            Z_parallel = 1 / Y_parallel;
            
        else
            error('Unknown configuration type.');
        end
        
        % Compute surface impedance and absorption
        Zs(i) = Z_parallel;
        Zs_norm(i) = Zs(i) / Z_air;
        Re(i) = real(Zs_norm(i));
        Im(i) = imag(Zs_norm(i));
        Reflect(i) = (Zs_norm(i) - 1) / (Zs_norm(i) + 1);
        alpha(i) = 1 - abs(Reflect(i)).^2;
    end
end
