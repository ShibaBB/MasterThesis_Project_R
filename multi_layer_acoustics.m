function [Reflect, Zs_norm, alpha, Re, Im] = multi_layer_acoustics(configuration, fibertype, parameters, freq, airProperties, coefficients)
    % configuration - 1: Fibers + Air, 2: Fibers + Fibers, 3: MPP + Fibers
    % fibertype - 1: Acrylic, 2: Silk, 3: Wool
    % parameters - Array containing layer properties
    % freq - Frequency vector (Hz)
    % airProperties - Struct containing air properties
    % coefficients - Fiber material coefficients in simplfied JCAL model
    
    % Extract air properties from airProperties
    Z_air = airProperties.impedance;
    v_sound = airProperties.speed_of_sound;
    j = 1i;
    
    % Initialize variables to store results
    Zs = zeros(1, length(freq));
    Reflect = zeros(1, length(freq));
    Zs_norm = zeros(1, length(freq));
    Re = zeros(1, length(freq));
    Im = zeros(1, length(freq));
    alpha = zeros(1, length(freq));
    
    % Loop over the frequency vector
    for i = 1:length(freq)
        % Current frequency
        current_freq = freq(i);
        
        % Initialize Transfer Matrix
        T = eye(2);
        
        % Loop through configuration to compute transfer matrices
        if configuration == 1  % Fibers + Air
            fiber_thickness = parameters(1);
            fiber_porosity = parameters(2);
            air_thickness = parameters(3);
            
            % Call jcal_s to calculate fiber impedance and wave number
            [~, ~, ~, ~, ~, Zc, k_fiber] = jcal_s(fibertype, fiber_thickness, fiber_porosity, coefficients, current_freq, airProperties);
            
            % Transfer matrix for fiber + air
            T_fiber = [cos(k_fiber * fiber_thickness), j * Zc * sin(k_fiber * fiber_thickness);
                       j * sin(k_fiber * fiber_thickness) / Zc, cos(k_fiber * fiber_thickness)];
            
            % Air layer transfer matrix
            k_air = 2 * pi * current_freq / v_sound;
            T_air = [cos(k_air * air_thickness), j * Z_air * sin(k_air * air_thickness);
                     j * sin(k_air * air_thickness) / Z_air, cos(k_air * air_thickness)];
            
            % Update total transfer matrix
            T = T * T_fiber * T_air;
        
        elseif configuration == 2  % Fibers + Fibers
            fiber1_thickness = parameters(1);
            fiber1_porosity = parameters(2);
            fiber2_thickness = parameters(3);
            fiber2_porosity = parameters(4);
            
            % Call jcal_s to calculate impedance and wave number for both fiber layers
            [~, ~, ~, ~, ~, Zc1, k_fiber1] = jcal_s(fibertype, fiber1_thickness, fiber1_porosity, coefficients, current_freq, airProperties);
            [~, ~, ~, ~, ~, Zc2, k_fiber2] = jcal_s(fibertype, fiber2_thickness, fiber2_porosity, coefficients, current_freq, airProperties);
            
            % Transfer matrix for fiber 1 and fiber 2
            T_fiber1 = [cos(k_fiber1 * fiber1_thickness), j * Zc1 * sin(k_fiber1 * fiber1_thickness);
                        j * sin(k_fiber1 * fiber1_thickness) / Zc1, cos(k_fiber1 * fiber1_thickness)];
            T_fiber2 = [cos(k_fiber2 * fiber2_thickness), j * Zc2 * sin(k_fiber2 * fiber2_thickness);
                        j * sin(k_fiber2 * fiber2_thickness) / Zc2, cos(k_fiber2 * fiber2_thickness)];
            
            % Update total transfer matrix
            T = T * T_fiber1 * T_fiber2;
        
        elseif configuration == 3  % MPP + Fibers
            fiber_thickness = parameters(1);
            fiber_porosity = parameters(2);
            mpp_theta = parameters(3);  % MPP perforation rate
            mpp_r = parameters(4);  % MPP micropore diameter
            mpp_t = parameters(5);  % MPP thickness
            
            % Call jcal_s to calculate fiber impedance and wave number
            [~, ~, ~, ~, ~, Zc, k_fiber] = jcal_s(fibertype, fiber_thickness, fiber_porosity, coefficients, current_freq, airProperties);
            
            % MPP layer impedance calculation (use MPP model)
            Z_MPP = MPP1(current_freq, mpp_theta, mpp_r, mpp_t, airProperties);
        
            % MPP layer transfer matrix
            T_MPP = [1, Z_MPP; 0, 1];
        
            % Fiber layer transfer matrix
            T_fiber = [cos(k_fiber * fiber_thickness), j * Zc * sin(k_fiber * fiber_thickness);
                      j * sin(k_fiber * fiber_thickness) / Zc, cos(k_fiber * fiber_thickness)];
    
            % Update total transfer matrix
            T = T * T_MPP * T_fiber;
        
        else
            error('Unknown configuration type.');
        end
        
        % Compute surface impedance and absorption for the current frequency
        Zs(i) = T(1,1) / T(2,1);
        Zs_norm(i) = Zs(i) / Z_air;
        Re(i) = real(Zs_norm(i));
        Im(i) = imag(Zs_norm(i));
        Reflect(i) = (Zs_norm(i) - 1) / (Zs_norm(i) + 1);
        alpha(i) = 1 - abs(Reflect(i)).^2;
    end
end
