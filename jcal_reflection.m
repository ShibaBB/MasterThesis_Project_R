function [Reflect, Zs_norm, alpha, Re, Im, Zc, k] = jcal_reflection(h, phi, sigma, alpha_infin, lambda, lambda_prime, k0_prime, freq, airProperties)
    % Extract air properties from the structure
    rho_air = airProperties.density_humid_air;  % Density of humid air (kg/m^3)
    air_Zc = airProperties.impedance;           % Characteristic impedance of air (kg/(m^2.s))
    eta = airProperties.eta;                   % Dynamic viscosity (kg/(m.s))
    gamma = airProperties.gamma;               % Ratio of specific heats
    pressure = airProperties.pressure;         % Air pressure (Pa)
    Pr = airProperties.Pr;                     % Prandtl number

    j = 1i;                 % Imaginary unit
    omega = 2 * pi * freq;  % Angular frequency (rad/s)

    % Check for valid input parameters (positive values)
    if any([phi, sigma, alpha_infin, lambda, lambda_prime, k0_prime] <= 0)
        error('All parameters phi, sigma, alpha_infin, lambda, lambda_prime, and k0_prime must be positive.');
    end

    % Dynamic density (rho_omega)
    rho_omega = (rho_air * alpha_infin ./ phi) .* (1 + (sigma * phi ./ (j * rho_air .* omega * alpha_infin)) .* ...
        (1 + 4 * j * omega * alpha_infin.^2 * eta * rho_air ./ (sigma^2 * lambda^2 * phi^2)).^0.5);

    % Dynamic modulus (K_omega)
    K_omega = (gamma * pressure / phi) .* (gamma - (gamma - 1) ./ ...
        (1 - (j * eta * phi) ./ (Pr * omega * k0_prime * rho_air) .* ...
        (1 + (4 * j * omega * k0_prime^2 * Pr * rho_air) ./ (eta * lambda_prime^2 * phi^2)).^0.5)).^-1;

    % Zwikker and Kosten for characteristic impedance (Zc)
    Zc = (rho_omega .* K_omega).^0.5;

    % Wave number (k)
    k = omega .* (rho_omega ./ K_omega).^0.5;

    % Surface impedance (Zs)
    Zs = Zc .* coth(j* k * h);  % Surface impedance

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
