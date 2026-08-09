function [Z] = MPP1(freq, theta, r, t, airProperties)
    % freq: frequency in Hz
    % theta: perforation rate (dimensionless)
    % r: diameter of micropores in meters
    % t: panel thickness in meters
    % airProperties: struct containing air properties such as air density (rho_air) and dynamic viscosity (eta)
    
    % Extract air properties from airProperties struct
    rho_air = airProperties.density_humid_air;  % Air density in kg/m^3
    eta = airProperties.eta;    % Dynamic viscosity of air in kg/(m*s)
    
    omega = 2 * pi * freq;  % Angular frequency
    j = sqrt(-1);  % Complex number
    
    % Correction length
    epsilon_e = 0.425 .* r .* (1 - 1.14 * sqrt(theta));
    
    % Surface resistance
    R_s = sqrt(eta * omega * rho_air / 2);
    
    % Impedance calculation
    Z = (t + 2 * epsilon_e) .* ((1 + j) * 4 .* R_s ./ (theta .* r) + (j * omega * rho_air) ./ theta);
end
