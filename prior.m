function logP = prior(params, lb, ub)
    % Small value to prevent log(0) issues
    eps_val = 1e-12;

    % Initialize log probability
    logP = 0;

    % Apply uniform priors using lb and ub
    for i = 1:length(params)
        if params(i) < lb(i) || params(i) > ub(i)
            logP = -Inf; % Parameter is out of bounds
            return;
        else
            uniform_prob = 1 / (ub(i) - lb(i)); % Uniform probability density
            logP = logP + log(uniform_prob + eps_val);
        end
    end
end
