%% Master Control: Run Inference for All Porosities of One Fiber
% 本脚本是总控脚本：
% 对同一种纤维材料的多个孔隙率样品，逐个进行贝叶斯反演和不确定性分析
fiberfolder = 'Wool';  % Only change fiber type here
porosity_list = {'92', '93', '94', '95', '96', '97', '98', '99'};  % All porosities to run

% 6 个待反演/输出参数名称：
% 前 5 个是 JCAL 材料参数，第 6 个是噪声标准差
param_names = {'Airflow Resistivity', 'Tortuosity', ...
               'Viscous Characteristic Length', 'Thermal Characteristic Length', ...
               'Static Thermal Permeability', 'Sigma Noise'};

% 用来保存每个孔隙率样品下，各参数的 R-hat 收敛诊断结果
% Store R-hat values for each porosity
all_rhat_results = zeros(length(porosity_list), length(param_names));

% 主循环：依次处理 92, 93, ... , 99 这些孔隙率文件夹
for p_idx = 1:length(porosity_list)
    porosityfolder = porosity_list{p_idx};
    fprintf('\n===== Running Inference for %s | Porosity %s =====\n', fiberfolder, porosityfolder);

    try
        %% 1 - Convert porosity folder to numerical porosity
        % 例如文件夹名 '92' 会被转成数值孔隙率 phi = 0.92
        phi = str2double(porosityfolder) / 100;

        %% 2 - Define File Paths
        % 以当前 MATLAB 工作目录作为项目根目录
        basePath = pwd;
        dataPath = fullfile(basePath, fiberfolder, porosityfolder);
        inputFile = fullfile(dataPath, 'R.txt');
        outputFile = fullfile(dataPath, 'converted_R.txt');

        % 如果实验反射系数文件不存在，就跳过当前样品
        if ~isfile(inputFile)
            warning('R.txt not found for %s - %s. Skipping.', fiberfolder, porosityfolder);
            continue;
        end

        %% 3 - Read and Clean Data
        % 原始 R.txt 使用小数逗号，先统一替换成 MATLAB 能识别的小数点
        rawData = fileread(inputFile);
        cleanedData = strrep(rawData, ',', '.');
        fid = fopen(outputFile, 'w'); fwrite(fid, cleanedData); fclose(fid);

        %% 4 - Load Converted Data
        % 读入转换后的实验数据
        data = readmatrix(outputFile);
        freq = data(:,1);

        % 仅保留 100 到 4950 Hz 频段的数据用于反演
        valid_idx = (freq >= 100) & (freq <= 4950);
        freq = freq(valid_idx);

        % 兼容两种文件格式：
        % 4 列时，第 2 列是反射系数实部，第 4 列是虚部
        % 3 列时，第 2 列是实部，第 3 列是虚部
        % movmean(...,2) 是一个简单的 2 点滑动平均，用来稍微平滑实验噪声
        if size(data, 2) == 4
            R_real_filtered = movmean(data(valid_idx,2), 2);
            R_imag_filtered = movmean(data(valid_idx,4), 2);
        elseif size(data, 2) == 3
            R_real_filtered = movmean(data(valid_idx,2), 2);
            R_imag_filtered = movmean(data(valid_idx,3), 2);
        else
            error('Unexpected column format');
        end

        % 将实验反射系数组装成复数形式
        R_exp = R_real_filtered + 1i * R_imag_filtered;

        % 再把复数拆成“实部向量 + 虚部向量”的长向量
        % 这样后续 log_likelihood 中可以统一做残差计算
        R_exp_concat = [real(R_exp); imag(R_exp)];

        %% 5 - Get Fluid Properties
        % 从对应 XML 文件中读取样品厚度和实验环境参数
        [thickness, temperature_K, pressure, rel_humidity, density_humid_air, Cp, Cv, eta, gamma, c, kappa, Pr] = getFluidProperties(fiberfolder, porosityfolder);

        % XML 中厚度单位是 mm，这里转成 m，供声学模型使用
        h = thickness * 1e-3;

        % 将正演模型需要的空气参数打包进结构体，便于传递
        airProperties = struct();
        airProperties.density_humid_air = density_humid_air;
        airProperties.speed_of_sound = c;
        airProperties.impedance = density_humid_air * c;
        airProperties.eta = eta;
        airProperties.gamma = gamma;
        airProperties.Pr = Pr;
        airProperties.pressure = pressure;

        %% 6 - Set Bayesian Parameters
        % 根据纤维类型和孔隙率生成参数上下界
        [lb, ub] = getFiberConstraints(fiberfolder, phi, airProperties);

        % 每个参数随机游走时使用的步长
        % 步长过大可能接受率很低，步长过小又会导致链移动很慢
        step_size = [20, 0.0005, 5e-8, 5e-8, 5e-10, 0.0005];  % reduced step sizes

        % 每条链采样 1000 次，共运行 3 条独立链
        n_samples = 1e+4;
        n_chains = 3;
        init_params_all = zeros(n_chains, length(lb));

        % 为每条链随机生成一个初始参数点
        % 这里故意避开边界，放在区间中间 60% 的区域内
        for chain = 1:n_chains
           % Avoid boundary by staying slightly away from edges
           init_params_all(chain, :) = lb + 0.2*(ub - lb) + 0.6*(ub - lb).*rand(1, length(lb));
        end

        %% 7 - Run MCMC
        % all_samples 的维度是：
        % 第 1 维 = 第几条链
        % 第 2 维 = 第几次迭代
        % 第 3 维 = 第几个参数
        all_samples = zeros(n_chains, n_samples, length(lb));
        for chain = 1:n_chains
            % 每条链单独运行一次 Metropolis-Hastings 采样
            all_samples(chain, :, :) = metropolis_hastings(n_samples, init_params_all(chain, :), h, freq, R_exp_concat, step_size, fiberfolder, phi, airProperties);
        end

        % 使用 Gelman-Rubin R-hat 诊断收敛情况
        % 如果各条链混合良好，R-hat 会接近 1
        % Compute R-hat diagnostics
        r_hat = zeros(1, length(lb));
        for param_idx = 1:length(lb)
            chain_samples = squeeze(all_samples(:, :, param_idx));
            chain_means = mean(chain_samples, 2);
            overall_mean = mean(chain_means);
            W = mean(var(chain_samples, 0, 2));
            B = n_samples * var(chain_means, 1);
            V_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B;
            r_hat(param_idx) = sqrt(V_hat / W);
        end
        all_rhat_results(p_idx, :) = r_hat;

        %% 8 - Combine Samples After Burn-in
        % 前 30% 视为 burn-in，认为这部分还在“从初值走向稳态”
        burn_in = round(0.3 * n_samples);
        samples_per_chain = n_samples - burn_in;

        % 去掉 burn-in 后，把 3 条链剩余样本拼成一个二维矩阵
        samples_after_burnin = reshape(all_samples(:, burn_in+1:end, :), n_chains * samples_per_chain, []);

        %% 9 - Compute Posterior Stats
        % 后验均值作为参数点估计，后验标准差作为不确定性量度
        estimated_params = mean(samples_after_burnin, 1);
        param_std = std(samples_after_burnin, 0, 1);

        %% 10 - Save Parameters
        % 将每个样品的参数均值、标准差和边界保存到 txt 文件
        filename = sprintf('%s_%s_params.txt', fiberfolder, porosityfolder);
        fileID = fopen(filename, 'w');
        if fileID == -1
            error('Could not open file for writing.');
        end
        fprintf(fileID, 'Inferred Parameters with Standard Deviation (SD) and Bounds\n');
        fprintf(fileID, 'Fiber Type: %s\n', fiberfolder);
        fprintf(fileID, 'Porosity: %s\n\n', porosityfolder);
        fprintf(fileID, '%-30s %-15s %-15s\n', 'Parameter Name', 'Value', 'SD');
        for i = 1:length(param_names)
            fprintf(fileID, '%-30s %.6e %.6e\n', param_names{i}, estimated_params(i), param_std(i));
        end
        fprintf(fileID, '\nParameter Constraints (Lower Bound, Upper Bound):\n');
        fprintf(fileID, '%-30s %-15s %-15s\n', 'Parameter Name', 'Lower Bound', 'Upper Bound');
        for i = 1:length(param_names)
            fprintf(fileID, '%-30s %.6e %.6e\n', param_names{i}, lb(i), ub(i));
        end
        fclose(fileID);

        %% 11 - Save Random Posterior Samples
        % 从后验样本中随机抽取 1000 组参数，便于后面做不确定性传播
        num_posterior_samples = 1000;
        rand_idx = randperm(size(samples_after_burnin, 1), num_posterior_samples);
        sampled_params = samples_after_burnin(rand_idx, :);

        % Save to file
        filename = sprintf('%s_%s_random_params.txt', fiberfolder, porosityfolder);
        fileID = fopen(filename, 'w');
        if fileID == -1
            error('Could not open random param file for writing.');
        end
        fprintf(fileID, 'Randomly Selected Posterior Samples\n');
        fprintf(fileID, 'Fiber Type: %s\n', fiberfolder);
        fprintf(fileID, 'Porosity: %s\n\n', porosityfolder);
        fprintf(fileID, '%-15s', 'Sample Index');
        for i = 1:length(param_names)
            fprintf(fileID, '%-20s', param_names{i});
        end
        fprintf(fileID, '\n');
        for i = 1:num_posterior_samples
            fprintf(fileID, '%-15d', i);
            fprintf(fileID, '%.6e ', sampled_params(i, :));
            fprintf(fileID, '\n');
        end
        fclose(fileID);

        %% --- Visualization 1: Trace Plots ---
        % Trace plot 用来看每条链随迭代如何移动，是否稳定、是否混合
        figure;
        tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        for i = 1:6
            nexttile;
            for ch = 1:n_chains
                samples = squeeze(all_samples(ch, :, i));
                if i == 6, samples = samples * 100; end
                plot(samples); hold on;
            end
            xline(burn_in, '--k', 'Burn-in', 'LabelOrientation', 'horizontal', 'FontSize', 10);  % 🔥 Added line
            yline(lb(i), '--r'); yline(ub(i), '--r');
            title(sprintf('%s Trace - %s %s', param_names{i}, fiberfolder, porosityfolder));
            xlabel('Iteration'); ylabel('Value'); grid on; box on;
        end

        %% --- Visualization 2: Histograms ---
        % 直方图用来查看每个参数的后验分布形状
        figure;
        tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        for i = 1:6
            nexttile;
            histogram(samples_after_burnin(:, i), 50);
            xline(estimated_params(i), '-k');
            xline(lb(i), '--r'); xline(ub(i), '--r');
            title(sprintf('%s Histogram - %s %s', param_names{i}, fiberfolder, porosityfolder));
            xlabel('Value'); ylabel('Frequency'); grid on; box on;
        end

        %% --- Visualization 3: Std/Mean ---
        % 用标准差/均值作为一个简化的相对不确定性指标
        figure;
        bar(param_std ./ estimated_params);
        set(gca, 'XTickLabel', param_names, 'XTickLabelRotation', 30);
        title(sprintf('Std/Mean Ratio - %s %s', fiberfolder, porosityfolder));
        ylabel('\sigma / \mu'); grid on;

        %% --- Visualization 4: Measured vs Predicted R and Alpha ---
        % 用后验均值参数做一次“代表性正演”
        % JCAL_O 使用固定空气常数；jcal_reflection 使用 XML 读取的实际空气参数
        [Zs_norm_O, Reflect_O, alpha_O] = JCAL_O(freq, h, phi, estimated_params(1), estimated_params(2), estimated_params(3), estimated_params(4), estimated_params(5));
        [R_predicted, ~, ~, ~, ~, ~, ~] = jcal_reflection(h, phi, estimated_params(1), estimated_params(2), estimated_params(3), estimated_params(4), estimated_params(5), freq, airProperties);
        alpha_exp = 1 - abs(R_exp).^2;
        alpha_predicted = 1 - abs(R_predicted).^2;

        % 对比实验与预测的反射系数实部
        figure;
        tiledlayout(2, 2);
        nexttile; scatter(freq, real(R_exp), 'r.'); hold on; plot(freq, real(R_predicted), 'b-', 'LineWidth', 2); grid on;
        title(['Real R - ', fiberfolder, ' ', porosityfolder]);
        xlabel('Freq'); ylabel('Real R');
        
        % 对比实验与预测的反射系数虚部
        nexttile; scatter(freq, imag(R_exp), 'r.'); hold on; plot(freq, imag(R_predicted), 'b-', 'LineWidth', 2); grid on;
        title(['Imag R - ', fiberfolder, ' ', porosityfolder]);
        xlabel('Freq'); ylabel('Imag R');

        % 对比实验吸声系数、后验均值预测吸声系数，以及固定空气参数版本的吸声系数
        nexttile([1 2]);
        scatter(freq, alpha_exp, 'r.'); hold on;
        plot(freq, alpha_predicted, 'b-', 'LineWidth', 2);
        plot(freq, alpha_O, '--k', 'LineWidth', 2);
        title(['Absorption - ', fiberfolder, ' ', porosityfolder]);
        xlabel('Frequency'); ylabel('\alpha'); grid on;

        %% --- Visualization 5: Random Posterior Predictive Samples ---
        % 用随机抽取的后验参数样本生成一簇预测曲线
        % 目的是展示参数不确定性会带来多大的预测散布
        figure;
        tiledlayout(2, 2);
        R_real_samples = zeros(length(freq), num_posterior_samples);
        R_imag_samples = zeros(length(freq), num_posterior_samples);
        alpha_post = zeros(length(freq), num_posterior_samples);
        for i = 1:num_posterior_samples
            % 对每一组后验样本单独做一次 JCAL 正演
            [R_s, ~, alpha_s] = jcal_reflection(h, phi, ...
                sampled_params(i,1), sampled_params(i,2), sampled_params(i,3), ...
                sampled_params(i,4), sampled_params(i,5), freq, airProperties);
            R_real_samples(:, i) = real(R_s);
            R_imag_samples(:, i) = imag(R_s);
            alpha_post(:, i) = alpha_s;
        end

        % 实验实部点云 + 后验预测实部曲线簇
        nexttile;
        scatter(freq, real(R_exp), 'r.'); hold on;
        plot(freq, R_real_samples, 'Color', [0.6 0.6 0.6 0.3]);
        title(['Real R Samples - ', fiberfolder, ' ', porosityfolder]);

        % 实验虚部点云 + 后验预测虚部曲线簇
        nexttile;
        scatter(freq, imag(R_exp), 'r.'); hold on;
        plot(freq, R_imag_samples, 'Color', [0.6 0.6 0.6 0.3]);
        title(['Imag R Samples - ', fiberfolder, ' ', porosityfolder]);

        % 实验吸声点云 + 后验预测吸声曲线簇
        nexttile([1 2]);
        scatter(freq, alpha_exp, 'r.'); hold on;
        plot(freq, alpha_post, 'Color', [0.6 0.6 0.6 0.3]);
        title(['Alpha Samples - ', fiberfolder, ' ', porosityfolder]);

    catch ME
        % 单个样品出错时不中断整个批处理，继续下一个孔隙率
        warning('❌ Error in %s | Porosity %s: %s', fiberfolder, porosityfolder, ME.message);
        continue;
    end
end

%% Display R-hat for all porosities
% 统一输出所有样品的收敛诊断结果
fprintf('\n===== R-hat Diagnostics for %s =====\n', fiberfolder);
for i = 1:length(porosity_list)
    fprintf('Porosity %s:\n', porosity_list{i});
    for j = 1:length(param_names)
        fprintf('  %s: %.4f\n', param_names{j}, all_rhat_results(i, j));
    end
end
