function logPosterior = posterior(params, h, freq, R_exp_concat, sigma, fiberfolder, porosityfolder, lb, ub)
    % Compute the prior using updated function that takes lb and ub
    logP = prior(params, lb, ub);

    % If prior is -Inf (invalid parameter range), immediately return -Inf
    if logP == -Inf
        logPosterior = -Inf;
        return;
    end

    % Compute the likelihood
    logL = likelihood(params, h, freq, R_exp_concat, sigma, fiberfolder, porosityfolder);

    % Combine the log prior and log likelihood to compute the log posterior
    logPosterior = logP + logL;
end
