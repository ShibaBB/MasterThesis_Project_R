function log_likelihood_value = log_likelihood(params, h, freq, R_exp_concat, fiberfolder, phi, airProperties)
    % 根据材料类型、孔隙率和空气参数，取得这组反演参数允许的上下界
    % Get the lower and upper bounds based on fiber type and porosity
    [lb, ub] = getFiberConstraints(fiberfolder, phi, airProperties);

    % 第 6 个参数不是材料参数，而是观测噪声标准差 sigma_noise
    % 在高斯似然里它必须为正，否则该参数组没有物理/统计意义
    % Extract sigma_noise from parameters and ensure positivity
    sigma_noise = params(6);
    if sigma_noise <= 0
        % 返回 -Inf 表示“这组参数绝不接受”
        log_likelihood_value = -Inf;
        return;
    end

    % 检查 6 个参数是否落在预设边界内
    % 这里等价于使用“有界均匀先验”：出界的点直接判为不合法
    % Check if parameters (excluding sigma_noise) are within the valid range
    if any(params(1:6) < lb) || any(params(1:6) > ub)
        log_likelihood_value = -Inf;  % Reject invalid parameters
        return;
    end

    % 依次取出前 5 个 JCAL 材料参数，便于后面调用正演模型
    % Extract individual parameters
    sigma = params(1);
    alpha_infin = params(2);
    lambda = params(3);
    lambda_prime = params(4);
    k0_prime = params(5);

    % 用 JCAL 正演模型根据当前候选参数计算“预测反射系数”
    % 输出里这里只关心 R_predicted，其他返回值用 ~ 忽略
    % Compute predicted reflection coefficient
    [R_predicted, ~, ~, ~, ~, ~, ~] = jcal_reflection(h, phi, sigma, alpha_infin, lambda, lambda_prime, k0_prime, freq, airProperties);

    % 将复反射系数拆成“实部 + 虚部”的长向量
    % 这样后面可以把实验值和预测值统一放进同一个残差表达式
    R_predicted_concat = [real(R_predicted); imag(R_predicted)];

    % 保护性检查：实验数据和模型预测必须是一一对应的同尺寸向量
    % Ensure matching size between R_exp_concat and R_predicted_concat
    if numel(R_exp_concat) ~= numel(R_predicted_concat)
        error('Shape mismatch between R_exp_concat and R_predicted_concat');
    end

    % 计算每个数据点的平方残差
    % 因为前面已经把实部和虚部拼接起来，所以这里同时在拟合幅值和相位信息
    % Compute residuals
    residuals = (R_exp_concat - R_predicted_concat).^2;

    % 假设每个残差都服从独立同分布的高斯噪声
    % 于是总的对数似然就是：
    % 1. 残差平方和项：残差越小，对数似然越大
    % 2. 归一化项：噪声 sigma_noise 越小，对拟合误差的容忍度越低
    % Compute log-likelihood (Gaussian noise assumption)
    N = numel(R_exp_concat);
    log_likelihood_value = -sum(residuals) / (2 * sigma_noise^2) - N * log(sigma_noise * sqrt(2 * pi));

end


% function log_likelihood_value = log_likelihood(params, h, freq, R_exp_concat, fiberfolder, phi, airProperties)
%     [lb, ub] = getFiberConstraints(fiberfolder, phi, airProperties);
%     tau = params(6);  % Formerly sigma_noise
% 
%     % Validity checks
%     if tau <= 0 || any(params < lb) || any(params > ub)
%         log_likelihood_value = -Inf;
%         return;
%     end
% 
%     % Extract parameters
%     sigma = params(1);
%     alpha_infin = params(2);
%     lambda = params(3);
%     lambda_prime = params(4);
%     k0_prime = params(5);
% 
%     % Model prediction
%     [R_predicted, ~, ~, ~, ~, ~, ~] = jcal_reflection(h, phi, sigma, alpha_infin, lambda, lambda_prime, k0_prime, freq, airProperties);
%     R_predicted_concat = [real(R_predicted); imag(R_predicted)];
%     residuals = R_exp_concat - R_predicted_concat;
% 
%     % Student’s t-distribution likelihood
%     nu = 5;  % degrees of freedom
%     log_likelihood_value = sum(gammaln((nu + 1)/2) - gammaln(nu/2) ...
%         - 0.5*log(nu*pi*tau^2) ...
%         - ((nu + 1)/2) * log(1 + (residuals.^2)/(nu*tau^2)));
% end
