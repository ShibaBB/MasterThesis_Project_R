%% parallel transfer matrix with closed elements
% surfaceRatio is an array
% phi is an array
function T_p = parallel_TM_CE(phi, optiSigma, optiAlphaInf, optiLambda, optiLambdaP, optiK0P, h, freqBand, surfaceRatio)
    N = length(freqBand);
    M = length(surfaceRatio);
    % initialize the admittance matrix for all elements
    Y_i_elements = zeros(2, 2, N, M);  
    T_p = zeros(2, 2, N);

    % collect all the admittance matrix into Y_i_elements
    for i = 1:M
        % admittance matrix
        Y_i_elements(:, :, :, i) = admittance_Matrix(phi(i), optiSigma(i), optiAlphaInf(i), optiLambda(i), optiLambdaP(i), optiK0P(i), h, freqBand);
    end

    for i = 1:N 
        sumOfWeightedY11 = 0;
        sumOfWeightedY12 = 0;
        sumOfWeightedY21 = 0;
        sumOfWeightedY22 = 0;
        for j = 1:M
            y_i = Y_i_elements(:, :, i, j);
            r = surfaceRatio(j);
            sumOfWeightedY11 = sumOfWeightedY11 + r * y_i(1, 1);
            sumOfWeightedY12 = sumOfWeightedY12 + r * y_i(1, 2);
            sumOfWeightedY21 = sumOfWeightedY21 + r * y_i(2, 1);
            sumOfWeightedY22 = sumOfWeightedY22 + r * y_i(2, 2);
            
        end
        T_p(:, :, i) = (-1 / sumOfWeightedY21) * [sumOfWeightedY22,     -1;
                        sumOfWeightedY22 * sumOfWeightedY11 - sumOfWeightedY12 * sumOfWeightedY21,      -1 * sumOfWeightedY11];

    end   
end

% admittance matrix
function Y_i = admittance_Matrix(phi, optiSigma, optiAlphaInf, optiLambda, optiLambdaP, optiK0P, h, freqBand)
    T_i = ElementTM_JCAL(phi, optiSigma, optiAlphaInf, optiLambda, optiLambdaP, optiK0P, h, freqBand);
    N = length(freqBand);

    Y_i = zeros(2, 2, N);

    for i = 1:N
        % transfer matrices for every frequency
        Y_i(:, :, i) = (1 / T_i(1, 2, i)) .* [T_i(2, 2, i),     T_i(2, 1, i) * T_i(1, 2, i) - T_i(2, 2, i) * T_i(1, 1, i);
                                              1,                -T_i(1, 1, i)];
    end

end

% Element Transfer Matrix
function T_i = ElementTM_JCAL(phi, optiSigma, optiAlphaInf, optiLambda, optiLambdaP, optiK0P, h, freqBand)

    % for JCAL Model
    [k_c, z_c] = JCAL_charater(phi, optiSigma, optiAlphaInf, optiLambda, optiLambdaP, optiK0P, freqBand);
    N = length(freqBand);

    T_i = zeros(2, 2, N); % initial 3D marix
    
    for i = 1:N
        % transfer matrices for every frequency
        T_i(:, :, i) = [cos(k_c(i) * h),               1i * z_c(i) * sin(k_c(i) * h);
                        1i * sin(k_c(i) * h) / z_c(i), cos(k_c(i) * h)];
    end
end

function [k_c, z_c] = JCAL_charater(phi, sigma, alpha_inf, lambda, lambda_prime, k0_prime, freq)
    %constants
    rho_0   = 1.204;            % air density
    P_0     = 101325;           % atmospheric pressure
    eta     = 1.846 * 10 ^ -5;  % air viscosity
    gamma   = 1.4;              % adiabatic constant
    Cp      = 1005;             % Specific heat capacity
    kappa   = 0.02596;          % Thermal conductivity
    
    %frequency
    omega = 2 * pi * freq;     % angular frequency
    
    % frequency-dependent effective density
    rho_tilde = (alpha_inf * rho_0 ./ phi) .* (1 + (sigma .* phi) ./ (1i .* omega * rho_0 * alpha_inf) .* ...
                sqrt(1 + 1i .* (4 * alpha_inf^2 * eta * rho_0 .* omega) ./ (sigma^2 * lambda^2 .* phi^2)));
    
    % compressibility
    K_tilde = (gamma * P_0 ./ phi) ./ (gamma - (gamma - 1) .* ...
              (1 - 1i .* phi * kappa ./ (k0_prime * Cp * rho_0 .* omega) .* ...
              sqrt(1 + 1i * 4 * k0_prime.^2 * Cp * rho_0 .* omega ./ (kappa * lambda_prime.^2 .* phi.^2))).^(-1));
    
    % characteristic impedance
    z_c = (rho_tilde .* K_tilde) .^ 0.5;

    % complex wave number
    k_c = omega .* (rho_tilde ./ K_tilde) .^ 0.5;
end
