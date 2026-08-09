function logL = likelihood(params, h, freq, R_exp_concat, fiberfolder, porosityfolder, airProperties)
    params = params(:)'; % Ensure row vector

    % Extract parameters
    sigma_f = params(1);  % Material-related parameter (sigma_f)
    alpha_infin = params(2);
    lambda = params(3);
    lambda_prime = params(4);
    k0_prime = params(5);
    sigma_noise  = params(6);

    phi = fiberParams.phi;  % Fixed porosity

    % Compute the predicted reflection coefficient using JCAL with fiberfolder and porosityfolder
    [R_model, ~, ~, ~, ~, ~, ~] = jcal_reflection(h, phi, sigma_f, alpha_infin, lambda, lambda_prime, k0_prime, freq, airProperties);
    R_model_real = real(R_model);
    R_model_imag = imag(R_model);

    % Normalize the real and imaginary parts of R_model
    R_model_real_norm = R_model_real / max(abs(R_model_real));
    R_model_imag_norm = R_model_imag / max(abs(R_model_imag));

    % Concatenate normalized real and imaginary parts into a single vector
    R_model_concat_norm = [R_model_real_norm; R_model_imag_norm]; 

    % Normalize the experimental reflection coefficient (R_exp_concat) in the same way
    R_exp_real = real(R_exp_concat);
    R_exp_imag = imag(R_exp_concat);

    % Normalize experimental data
    R_exp_real_norm = R_exp_real / max(abs(R_exp_real));
    R_exp_imag_norm = R_exp_imag / max(abs(R_exp_imag));

    % Concatenate normalized real and imaginary parts of R_exp_concat
    R_exp_concat_norm = [R_exp_real_norm; R_exp_imag_norm]; 

    % Ensure the shapes of R_exp_concat_norm and R_model_concat are compatible
    if numel(R_exp_concat_norm) ~= numel(R_model_concat_norm)
        error('Shape mismatch between R_exp_concat_norm and R_model_concat');
    end

    % Compute squared magnitude difference between normalized experimental and model data
    residuals = abs(R_exp_concat_norm - R_model_concat_norm).^2;

    % Compute log-likelihood (Gaussian noise assumption)
    N = numel(R_exp_concat_norm);
    logL = -sum(residuals) / (2 * sigma_noise^2) - N * log(sigma_noise * sqrt(2 * pi));

end


% function logL = likelihood(params, h, freq, R_exp_concat, fiberfolder, porosityfolder, airProperties)
%     params = params(:)';  % Ensure row vector
% 
%     % Extract parameters
%     sigma_f = params(1);
%     alpha_infin = params(2);
%     lambda = params(3);
%     lambda_prime = params(4);
%     k0_prime = params(5);
%     tau = params(6);  % Previously called sigma_noise
% 
%     phi = fiberParams.phi;
% 
%     % JCAL model
%     [R_model, ~, ~, ~, ~, ~, ~] = jcal_reflection(h, phi, sigma_f, alpha_infin, lambda, lambda_prime, k0_prime, freq, airProperties);
%     R_model_concat = [real(R_model); imag(R_model)];
%     R_exp_concat_vec = [real(R_exp_concat); imag(R_exp_concat)];
% 
%     % Residuals
%     residuals = R_exp_concat_vec - R_model_concat;
% 
%     % Student’s t-distribution log-likelihood
%     nu = 5;  % degrees of freedom (tune this or make it a parameter)
%     logL = sum(gammaln((nu + 1)/2) - gammaln(nu/2) ...
%          - 0.5*log(nu*pi*tau^2) ...
%          - ((nu + 1)/2) * log(1 + (residuals.^2)/(nu*tau^2)));
% end