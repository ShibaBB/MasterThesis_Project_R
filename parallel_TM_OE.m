%% parallel transfer matrix
% for the situation that the heights of elements are different
function T_p = parallel_TM_OE(phi, optiSigma, optiAlphaInf, optiLambda, optiLambdaP, optiK0P, h, freqBand, surfaceRatio, maxHeight)
    N = length(freqBand);
    M = length(surfaceRatio);
    % initialize the admittance matrix for all elements
    Y_i_elements = zeros(2, 2, N, M);  
    T_p = zeros(2, 2, N);
    tolerance = 0.002;

    % collect all the admittance matrix into Y_i_elements

    for i = 1:N 
        sumOfWeightedYi11 = 0;
        sumOfWeightedYi12 = 0;
        sumOfWeightedYi21 = 0;
        sumOfWeightedYi22 = 0;
        sumOfWeightedOE = 0;
        for j = 1:M
            r = surfaceRatio(j);
            if h(j) >= maxHeight - tolerance && h(j) <= maxHeight + tolerance % close end
                T_i = ElementTM_JCAL(phi(j), optiSigma(j), optiAlphaInf(j), optiLambda(j), optiLambdaP(j), optiK0P(j), h(j), freqBand(i));
                Y_i_elements(:, :, i, j) = admittance_Matrix(T_i, freqBand(i));
                y_i = Y_i_elements(:, :, i, j);

            else % open end
                airGap = maxHeight - h(j);
                T_air_matrix = airSpaceTM(freqBand(i), airGap);
                T_i = ElementTM_JCAL(phi(j), optiSigma(j), optiAlphaInf(j), optiLambda(j), optiLambdaP(j), optiK0P(j), h(j), freqBand(i));
                T_total = T_i * T_air_matrix;
                Y_i_elements(:, :, i, j) = admittance_Matrix(T_total, freqBand(i));
                y_i = Y_i_elements(:, :, i, j);

            end
            sumOfWeightedYi11 = sumOfWeightedYi11 + r * y_i(1, 1);
            sumOfWeightedYi12 = sumOfWeightedYi12 + r * y_i(1, 2);
            sumOfWeightedYi21 = sumOfWeightedYi21 + r * y_i(2, 1);
            sumOfWeightedYi22 = sumOfWeightedYi22 + r * y_i(2, 2);
        end
        T_p(:, :, i) = (-1 / sumOfWeightedYi21) * [sumOfWeightedYi22,     -1;
                        sumOfWeightedYi22 * (sumOfWeightedYi11 - sumOfWeightedOE) - sumOfWeightedYi12 * sumOfWeightedYi21,     sumOfWeightedOE - sumOfWeightedYi11];

    end   
end

% admittance matrix
function Y_i = admittance_Matrix(T_i, freqBand)
    % T_i = ElementTM_JCAL(phi, optiSigma, optiAlphaInf, optiLambda, optiLambdaP, optiK0P, h, freqBand);
    N = length(freqBand);

    Y_i = zeros(2, 2, N);

    for i = 1:N
        % transfer matrices for every frequency
        Y_i(:, :, i) = (1 / T_i(1, 2, i)) .* [T_i(2, 2, i),     T_i(2, 1, i) * T_i(1, 2, i) - T_i(2, 2, i) * T_i(1, 1, i);
                                              1,                -T_i(1, 1, i)];
    end

end

% Element Transfer Matrix
function T_i = ElementTM_JCAL(phi, optiSigma, optiAlphaInf, optiLambda, optiLambdaP, optiK0P, h, freq)

    % for JCAL Model
    [k_c, z_c] = JCAL_charater(phi, optiSigma, optiAlphaInf, optiLambda, optiLambdaP, optiK0P, freq);
    N = length(freq);

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
