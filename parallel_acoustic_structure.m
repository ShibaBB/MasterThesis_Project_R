function [Reflect, Zs_norm, alpha, Re, Im] = parallel_acoustic_structure(configuration, fibertype, parameters, freq, airProperties, coefficients, area_ratios)
    % Compute sound absorption using Admittance Sum Method (ASM)
    % Based on Verdière et al. (2014), applying the correct surface admittance approach.
    
    % Extract air properties
    Z_air = airProperties.impedance;
    v_sound = airProperties.speed_of_sound;
    j = 1i;  % Imaginary unit
    
    % Initialize results
    num_freq = length(freq);
    Reflect = zeros(1, num_freq);
    Zs_norm = zeros(1, num_freq);
    Re = zeros(1, num_freq);
    Im = zeros(1, num_freq);
    alpha = zeros(1, num_freq);
    
    % Loop over frequency range
    for i = 1:num_freq
        current_freq = freq(i);
        k_air = 2 * pi * current_freq / v_sound;
        
        % Initialize admittance sum
        YG = 0;
        
        % Process different configurations
        if configuration == 1  % Fibers + Air
            fiber_thickness = parameters(1);
            fiber_porosity = parameters(2);
            air_thickness = parameters(3);
            
            [~, ~, ~, ~, ~, Zc_fiber, k_fiber] = jcal_s(fibertype, fiber_thickness, fiber_porosity, coefficients, current_freq, airProperties);
            Y_fiber = (j * sin(k_fiber * fiber_thickness)) ./ (Zc_fiber .* cos(k_fiber * fiber_thickness));

            Y_air = (j * sin(k_air * air_thickness)) ./ (Z_air .* cos(k_air * air_thickness));
            
            % Compute global admittance
            YG = area_ratios(1) * Y_fiber + area_ratios(2) * Y_air;
            
        elseif configuration == 2  % Fibers + Fibers
            fiber1_thickness = parameters(1);
            fiber1_porosity = parameters(2);
            fiber2_thickness = parameters(3);
            fiber2_porosity = parameters(4);
            
            [~, ~, ~, ~, ~, Zc_fiber1, k_fiber1] = jcal_s(fibertype, fiber1_thickness, fiber1_porosity, coefficients, current_freq, airProperties);
            [~, ~, ~, ~, ~, Zc_fiber2, k_fiber2] = jcal_s(fibertype, fiber2_thickness, fiber2_porosity, coefficients, current_freq, airProperties);

            Y_fiber1 = (j * sin(k_fiber1 * fiber1_thickness)) ./ (Zc_fiber1 .* cos(k_fiber1 * fiber1_thickness));
            Y_fiber2 = (j * sin(k_fiber2 * fiber2_thickness)) ./ (Zc_fiber2 .* cos(k_fiber2 * fiber2_thickness));
            
            YG = area_ratios(1) * Y_fiber1 + area_ratios(2) * Y_fiber2;
            
        elseif configuration == 3  % MPP + Fibers with Air Cavity
            fiber3_thickness = parameters(1);
            fiber3_porosity = parameters(2);
            mpp_theta = parameters(3);
            mpp_r = parameters(4);
            mpp_t = parameters(5);
            air_cavity_thickness = parameters(6);
            
            [~, ~, ~, ~, ~, Zc_fiber3, k_fiber3] = jcal_s(fibertype, fiber3_thickness, fiber3_porosity, coefficients, current_freq, airProperties);

            Y_fiber3 = (j * sin(k_fiber3 * fiber3_thickness)) ./ (Zc_fiber3 .* cos(k_fiber3 * fiber3_thickness));
         
            Z_MPP = MPP1(current_freq, mpp_theta, mpp_r, mpp_t, airProperties);   
            T_MPP = [1, Z_MPP; 0, 1];
            T_air = [cos(k_air * air_cavity_thickness), j * Z_air * sin(k_air * air_cavity_thickness); j * sin(k_air * air_cavity_thickness) / Z_air, cos(k_air * air_cavity_thickness)];
            T = T_MPP * T_air;

            Y_MPP_air=T(2,1)/T(1,1);
            
            YG = area_ratios(1) * Y_fiber3 + area_ratios(2) * Y_MPP_air;
        end
        
        alpha(i) = 1 - abs((1-YG * Z_air)/(1+YG * Z_air))^2;
        Reflect(i) = (1 - YG * Z_air) / (1 + YG * Z_air);
        Zs_norm(i) = (1 + Reflect(i)) / (1 - Reflect(i));
        Re(i) = real(Zs_norm(i));
        Im(i) = imag(Zs_norm(i));

    end
end
