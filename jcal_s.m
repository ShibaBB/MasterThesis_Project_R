function [Reflect, Zs_norm, alpha, Re, Im, Zc, k] = jcal_s(fibertype, h, phi, coefficients, freq, airProperties)
    % Extract air properties from the structure
    rho_air = airProperties.density_humid_air;  % Density of humid air (kg/m^3)
    air_Zc = airProperties.impedance;           % Characteristic impedance of air (kg/(m^2.s))
    eta = airProperties.eta;                    % Dynamic viscosity (kg/(m.s))
    gamma = airProperties.gamma;                % Ratio of specific heats
    pressure = airProperties.pressure;          % Air pressure (Pa)
    Pr = airProperties.Pr;                      % Prandtl number
    
    j = 1i;                 % Imaginary unit
    omega = 2 * pi * freq;  % Angular frequency (rad/s)
    
    % Select material coefficients based on fibertype
    switch fibertype
        case 1 % Acrylic fiber
            a = coefficients.a1;
            b = coefficients.b1;
            c = coefficients.c1;
            d = coefficients.d1;
            e = coefficients.e1;
        case 2 % Silk fiber
            a = coefficients.a2;
            b = coefficients.b2;
            c = coefficients.c2;
            d = coefficients.d2;
            e = coefficients.e2;
        case 3 % Wool fiber
            a = coefficients.a3;
            b = coefficients.b3;
            c = coefficients.c3;
            d = coefficients.d3;
            e = coefficients.e3;
        otherwise
            error('Invalid fibertype. Choose 1 (Acrylic), 2 (Silk), or 3 (Wool).');
    end
    
    % Compute material properties
    sigma = a * (1 - phi) ./ (-0.640 * log(1 - phi) + 0.263 - phi); % Airflow resistivity
    alpha_infin = 1 - b*log(phi); % Tortuosity
    lambda = c * sqrt(((1 - b * log(phi)) .* (-0.64 * log(1 - phi) - phi + 0.263)) ./ ((1 - phi) .* phi)) * 1e-6; % Viscous characteristic length
    lambda_prime = d * sqrt(((1 - b * log(phi)) .* (-0.64 * log(1 - phi) - phi + 0.263)) ./ ((1 - phi) .* phi)) * 1e-6; % Thermal characteristic length
    k0_prime = e * (phi.^2 ./ (1 - phi)) * 1e-10; % Thermal permeability
    
    % Check for valid input parameters (positive values)
    if any([phi, sigma, alpha_infin, lambda, lambda_prime, k0_prime] <= 0)
        error('All parameters phi, sigma, alpha_infin, lambda, lambda_prime, and k0_prime must be positive.');
    end
    
    % Dynamic density (rho_omega)
    % rho_omega = (rho_air * alpha_infin ./ phi) .* (1 + (sigma .* phi ./ (j * rho_air .* omega .* alpha_infin)) .* ...
    %     (1 + 4 * j * omega * alpha_infin.^2 * eta * rho_air ./ (sigma^2 * lambda^2 * phi^2)).^0.5);

    [Phi, Omega] = meshgrid(phi, omega); % Expand phi and omega to match dimensions

    rho_omega = (rho_air * alpha_infin ./ Phi) .* (1 + (sigma .* Phi ./ (j * rho_air .* Omega .* alpha_infin)) .* ...
        (1 + 4 * j .* Omega .* alpha_infin.^2 .* eta .* rho_air ./ (sigma.^2 .* lambda.^2 .* Phi.^2)).^0.5);

    % Dynamic modulus (K_omega)
    % K_omega = (gamma * pressure / phi) .* (gamma - (gamma - 1) ./ ...
    %     (1 - (j * eta * phi) ./ (Pr * omega * k0_prime * rho_air) .* ...
    %     (1 + (4 * j * omega * k0_prime^2 * Pr * rho_air) ./ (eta * lambda_prime^2 * phi^2)).^0.5)).^-1;

    K_omega = (gamma * pressure ./ Phi) .* (gamma - (gamma - 1) ./ ...
        (1 - (j * eta .* Phi) ./ (Pr .* Omega .* k0_prime .* rho_air) .* ...
        (1 + (4 * j .* Omega .* k0_prime.^2 .* Pr .* rho_air) ./ (eta .* lambda_prime.^2 .* Phi.^2)).^0.5)).^-1;
    
    % Characteristic impedance (Zc)
    Zc = sqrt(rho_omega .* K_omega);
    
    % Wave number (k)
    k = omega .* sqrt(rho_omega ./ K_omega);
    
    % Surface impedance (Zs)
    Zs = Zc .* coth(j * k .* h);  % Surface impedance
    
    % Normalized surface impedance (Zs_norm)
    Zs_norm = Zs / air_Zc;  % Normalized by the air impedance
    
    % Reflection coefficient (R = (Zs - Zc) / (Zs + Zc))
    Reflect = (Zs_norm - 1) ./ (Zs_norm + 1);
    
    % Surface impedance components (real and imaginary parts)
    Re = real(Zs) / air_Zc;  % Normalized real part
    Im = imag(Zs) / air_Zc;  % Normalized imaginary part
    
    % Predicted sound absorption (alpha = 1 - |R|^2)
    alpha = 1 - abs(Reflect).^2;
end
