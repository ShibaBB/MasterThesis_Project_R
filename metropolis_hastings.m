function samples = metropolis_hastings(n_samples, init_params, h, freq, R_exp_concat, step_size, fiberfolder, phi, airProperties)
    % current_params 表示“这条链当前所在的参数位置”
    % 一开始先用用户给定的初值作为链的起点
    % Initialize parameters and setup
    current_params = init_params;

    % 对当前这组参数计算一次对数似然，作为后续比较的基准分数
    current_log_likelihood = log_likelihood(current_params, h, freq, R_exp_concat, fiberfolder, phi, airProperties);

    % samples 是这条 MCMC 链最终输出的样本矩阵：
    % 行 = 第几次迭代
    % 列 = 第几个参数
    % Store the parameter samples
    samples = zeros(n_samples, length(init_params));

    % 第 1 个样本就是初始化位置
    samples(1, :) = current_params;  

    % 取得这类材料、该孔隙率下允许的参数上下界
    % 后面每次提议新参数时，都不允许跑出这个物理范围
    % Get constraints for first 6 parameters
    [lb, ub] = getFiberConstraints(fiberfolder, phi, airProperties);

    % 每完成 1% 的采样打印一次进度
    % Define progress reporting interval
    report_interval = floor(n_samples / 100);

    % 从第 2 次样本开始迭代，因为第 1 次已经被初始化占用了
    % MCMC loop
    for i = 2:n_samples
        % 以“当前参数 + 高斯随机扰动”的方式生成一个新候选点
        % step_size 控制每个参数一步最多通常会跳多远
        % Propose new parameters by adding random noise
        proposed_params = current_params + step_size .* randn(size(current_params));

        % 如果某个候选参数越界，就把它截回边界内
        % 这是一种简单的有界随机游走提议机制
        % Ensure parameters stay within bounds
        proposed_params = max(lb, min(ub, proposed_params));

        % 计算新候选点的对数似然
        % 这里本质上是在问：这组新参数解释实验数据的能力如何？
        % Compute log likelihood for proposed parameters
        proposed_log_likelihood = log_likelihood(proposed_params, h, freq, R_exp_concat, fiberfolder, phi, airProperties);

        % 计算接受率：
        % 如果新点更好（对数似然更大），这个值会 > 1，相当于必定接受
        % 如果新点更差，这个值会落在 0 到 1 之间，表示“有一定概率仍然接受”
        % Calculate acceptance ratio
        acceptance_ratio = exp(proposed_log_likelihood - current_log_likelihood);

        % 用一个 0~1 的随机数决定是否采纳新点
        % 接受：链移动到 proposed_params
        % 拒绝：链停留在 current_params
        % Accept or reject
        if rand() < acceptance_ratio
            current_params = proposed_params;
            current_log_likelihood = proposed_log_likelihood;
        end
        
        % 无论这一步接受还是拒绝，都把“当前所在位置”记成第 i 个样本
        % 如果拒绝了，这一行会和上一行相同
        % Store the sample
        samples(i, :) = current_params;

        % 打印当前采样进度
        if mod(i, report_interval) == 0
          fprintf('\rProgress: %.0f%% (%d/%d samples)', (i / n_samples) * 100, i, n_samples);
        end

    end
end
