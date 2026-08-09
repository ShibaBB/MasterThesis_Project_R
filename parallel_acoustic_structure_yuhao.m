function [Reflect, Zs_norm, alpha, Re, Im] = parallel_acoustic_structure(configuration, fibertype, parameters, freq, airProperties, coefficients, area_ratios)
    % Compute sound absorption, reflection coefficient, and surface impedance
    % using the Transfer Matrix Method for parallel assemblies.
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
    j = 1i;
    
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
        sum_r_y11 = 0;
        sum_r_y12 = 0;
        sum_r_y21 = 0;
        sum_r_y22 = 0;
        
        for j = 1:length(area_ratios)
            if configuration == 1  % Fibers + Air
                fiber_thickness = parameters(1);
                fiber_porosity = parameters(2);
                air_thickness = parameters(3);
                
                [~, ~, ~, ~, ~, Z_fiber, k_fiber] = jcal_s(fibertype, fiber_thickness, fiber_porosity, coefficients, current_freq, airProperties);
                T_fiber = [cos(k_fiber * fiber_thickness), j * Z_fiber * sin(k_fiber * fiber_thickness);
                          j * sin(k_fiber * fiber_thickness) / Z_fiber, cos(k_fiber * fiber_thickness)];
                
                k_air = 2 * pi * current_freq / v_sound;
                T_air = [cos(k_air * air_thickness), j * Z_air * sin(k_air * air_thickness);
                         j * sin(k_air * air_thickness) / Z_air, cos(k_air * air_thickness)];
                
                Y_fiber = inv([T_fiber(2,2), -T_fiber(1,2); -T_fiber(2,1), T_fiber(1,1)]);
                Y_air = inv([T_air(2,2), -T_air(1,2); -T_air(2,1), T_air(1,1)]);
                
                Y_i = area_ratios(1) * Y_fiber + area_ratios(2) * Y_air;
            
            elseif configuration == 2  % Fibers + Fibers
                fiber1_thickness = parameters(1);
                fiber1_porosity = parameters(2);
                fiber2_thickness = parameters(3);
                fiber2_porosity = parameters(4);
                
                [~, ~, ~, ~, ~, Z_fiber1, k_fiber1] = jcal_s(fibertype, fiber1_thickness, fiber1_porosity, coefficients, current_freq, airProperties);
                [~, ~, ~, ~, ~, Z_fiber2, k_fiber2] = jcal_s(fibertype, fiber2_thickness, fiber2_porosity, coefficients, current_freq, airProperties);
                
                T_fiber1 = [cos(k_fiber1 * fiber1_thickness), j * Z_fiber1 * sin(k_fiber1 * fiber1_thickness);
                            j * sin(k_fiber1 * fiber1_thickness) / Z_fiber1, cos(k_fiber1 * fiber1_thickness)];
                T_fiber2 = [cos(k_fiber2 * fiber2_thickness), j * Z_fiber2 * sin(k_fiber2 * fiber2_thickness);
                            j * sin(k_fiber2 * fiber2_thickness) / Z_fiber2, cos(k_fiber2 * fiber2_thickness)];
                
                Y_fiber1 = inv([T_fiber1(2,2), -T_fiber1(1,2); -T_fiber1(2,1), T_fiber1(1,1)]);
                Y_fiber2 = inv([T_fiber2(2,2), -T_fiber2(1,2); -T_fiber2(2,1), T_fiber2(1,1)]);
                
                Y_i = area_ratios(1) * Y_fiber1 + area_ratios(2) * Y_fiber2;
            
            elseif configuration == 3  % MPP + Fibers
                fiber_thickness = parameters(1);
                fiber_porosity = parameters(2);
                mpp_theta = parameters(3);
                mpp_r = parameters(4);
                mpp_t = parameters(5);
                
                [~, ~, ~, ~, ~, Z_fiber, k_fiber] = jcal_s(fibertype, fiber_thickness, fiber_porosity, coefficients, current_freq, airProperties);
                Z_MPP = MPP1(current_freq, mpp_theta, mpp_r, mpp_t, airProperties);
                
                T_fiber = [cos(k_fiber * fiber_thickness), j * Z_fiber * sin(k_fiber * fiber_thickness);
                          j * sin(k_fiber * fiber_thickness) / Z_fiber, cos(k_fiber * fiber_thickness)];
                T_MPP = [1, Z_MPP; 0, 1];
                
                Y_fiber = inv([T_fiber(2,2), -T_fiber(1,2); -T_fiber(2,1), T_fiber(1,1)]);
                Y_MPP = inv(T_MPP);
                
                Y_i = area_ratios(1) * Y_fiber + area_ratios(2) * Y_MPP;
            end
            
            sum_r_y11 = sum_r_y11 + Y_i(1,1);
            sum_r_y12 = sum_r_y12 + Y_i(1,2);
            sum_r_y21 = sum_r_y21 + Y_i(2,1);
            sum_r_y22 = sum_r_y22 + Y_i(2,2);
        end
        
        T_p = (-1 / sum_r_y21) * [sum_r_y22, -1; sum_r_y22 * sum_r_y11 - sum_r_y12 * sum_r_y21, -sum_r_y11];
        
        Zs(i) = sqrt(T_p(1,2) / T_p(2,1));
        Zs_norm(i) = Zs(i) / Z_air;
        Re(i) = real(Zs_norm(i));
        Im(i) = imag(Zs_norm(i));
        Reflect(i) = (Zs_norm(i) - 1) / (Zs_norm(i) + 1);
        alpha(i) = max(0, min(1, 1 - abs(Reflect(i)).^2));
    end
end
