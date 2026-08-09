# 解读不确定性量化脚本

> 我的导师给了我一系列代码，是关于如何计算材料的声学参数的，我现在需要解读Uncertainty_quantification.m文件，你帮我解读一下这个文件是干什么的，里面用到的所有子文件你都可以在目录下找到

<details><summary>5 previous messages</summary>

> 我先定位 `Uncertainty_quantification.m`，读它的主流程，再把它调用到的本地函数和输入文件一起梳理出来，最后给你一个按“目的-流程-每个子文件作用”的解读。
>
> 我已经定位到主文件。下一步我会把 `Uncertainty_quantification.m` 逐段读出来，并反查它显式调用的 `.m` 子函数，这样解释不会停留在表面。
>
> 主文件已经很清楚地暴露了核心依赖：`getFluidProperties`、`getFiberConstraints`、`metropolis_hastings`、`jcal_reflection`、`JCAL_O`。我现在把这些子文件读完，并顺带确认它们继续调用了哪些函数。
>
> MCMC 内层还依赖 `log_likelihood.m`，这个函数决定“什么样的参数算拟合得好”。我把它也读出来，并顺带区分哪些相关文件是“真的被调用”，哪些只是同一套贝叶斯代码里没被这个脚本用到。
>
> 我还要确认它读入的原始实验数据长什么样，因为这决定脚本前半段到底在做“格式清洗”还是“物理处理”。我会抽一组 `Wool/92` 的 `R.txt` 和 XML 看输入结构。
</details>

[Uncertainty_quantification.m](<./Uncertainty_quantification.m:1>) 的作用，可以概括成一句话：它用实验测得的反射系数 `R.txt`，结合 JCAL 多孔材料声学模型，做一轮基于 MCMC 的贝叶斯反演，从而同时得到“材料声学参数的估计值”和“这些参数的不确定性”。

它现在被写成“批处理”脚本：顶部把 `fiberfolder` 固定成 `Wool`，然后依次跑 92% 到 99% 孔隙率。每个孔隙率都会做这些事：
1. 读 [Wool/92/R.txt](<./Wool/92/R.txt:1>) 这种实验文件，先把欧洲小数逗号 `,` 转成 MATLAB 能读的 `.`。
2. 取 100 到 4950 Hz 的频段，并对实部/虚部做一个 2 点滑动平均。
3. 读对应的 XML，例如 [Wool/92/Wool_92.aed1001.xml](<./Wool/92/Wool_92.aed1001.xml:1>)，提取样品厚度、温度、气压、湿度，再计算湿空气的密度、粘度、声速、Prandtl 数等流体参数。
4. 给 6 个待估参数设置上下界：前 5 个是 JCAL 材料参数，第 6 个 `Sigma Noise` 是噪声尺度，不是材料本体参数。
5. 用 3 条 MCMC 链、每条 1000 个样本做 Metropolis-Hastings 采样。
6. 算每个参数的 `R-hat`，检查链是否收敛到接近 1。
7. 丢弃前 30% burn-in，把后验样本合并，输出均值和标准差。
8. 保存两个结果文件：`Wool_92_params.txt` 这类“均值+标准差+边界”，以及 `Wool_92_random_params.txt` 这类“随机抽取的后验样本”。
9. 画 5 类图：trace plot、后验直方图、相对不确定度 `std/mean`、实验/预测反射系数对比、后验预测样本云图。

实际调用链是这条：

```text
Uncertainty_quantification.m
├─ getFluidProperties.m
├─ getFiberConstraints.m
├─ metropolis_hastings.m
│  ├─ getFiberConstraints.m
│  └─ log_likelihood.m
│     ├─ getFiberConstraints.m
│     └─ jcal_reflection.m
├─ jcal_reflection.m
└─ JCAL_O.m
```

这些子文件分别干什么：
- [getFluidProperties.m](<./getFluidProperties.m:1>)：从 XML 里读实验环境和厚度，并计算湿空气物性。
- [getFiberConstraints.m](<./getFiberConstraints.m:1>)：根据纤维类型和孔隙率，给参数生成物理上合理的上下界。
- [metropolis_hastings.m](<./metropolis_hastings.m:1>)：真正执行 MCMC 采样。
- [log_likelihood.m](<./log_likelihood.m:1>)：定义“拟合好坏”。它把模型预测的反射系数和实验反射系数比较，差越小，似然越大。
- [jcal_reflection.m](<./jcal_reflection.m:1>)：JCAL 正演模型。输入材料参数和空气参数，输出反射系数、表面阻抗、吸声系数。
- [JCAL_O.m](<./JCAL_O.m:1>)：也是 JCAL 正演，但空气参数用固定常数，不用 XML 里的实际环境值。这个脚本里主要拿它画一条对照吸声曲线。

有几点需要特别注意：
- 这份脚本虽然叫“不确定性量化”，本质上是在做“后验分布采样”。不确定性主要体现在标准差、随机后验样本、以及 posterior predictive 图上。
- 同目录的 [likelihood.m](<./likelihood.m:1>)、[prior.m](<./prior.m:1>)、[posterior.m](<./posterior.m:1>) 这次并没有被这个脚本调用，不用把注意力放在那里。
- 从代码看，它相当于用了“有边界的均匀先验”：参数只允许落在 `lb` 和 `ub` 之间，超出就直接拒绝。
- 第 21 行和 `getFluidProperties.m` 第 11 行都把路径硬编码成了 `/Users/tao/Documents/MATLAB/Fibers`。这说明它原本是给导师或同组同学的 Mac 环境写的，在你这台机器上直接运行前大概率要先改路径。
- 输出的 `*_params.txt` 和 `*_random_params.txt` 是写到当前工作目录，不是写回各自的 `Wool/92` 子目录。

如果你愿意，我下一步可以继续帮你做一版“逐行解读”，把 [Uncertainty_quantification.m](<./Uncertainty_quantification.m:1>) 按代码块一段一段翻成中文注释。

> 好的，谢谢

不客气。

如果你下一步要继续看这个项目，建议优先连着读这几个文件：
[jcal_reflection.m](<./jcal_reflection.m:1>)、[log_likelihood.m](<./log_likelihood.m:1>)、[metropolis_hastings.m](<./metropolis_hastings.m:1>)。这三个基本就是“物理模型 + 拟合目标 + 采样算法”的核心。

如果你要，我可以下一条直接给你做：
1. `Uncertainty_quantification.m` 逐行中文注释版
2. 这个文件的整体流程图
3. 6 个参数各自的物理意义解释

> 帮我全都做一下，谢谢

<details><summary>1 previous message</summary>

> 我按你要的三部分整理：先做 `Uncertainty_quantification.m` 的逐段逐行中文解读，再给出完整流程图，最后解释 6 个反演参数的物理意义和它们对吸声/反射的影响。
</details>

主文件是 [Uncertainty_quantification.m](<./Uncertainty_quantification.m:1>)。它的核心任务是：对每个孔隙率样品，用实验反射系数 `R.txt` 去反演 5 个 JCAL 材料参数，加上 1 个噪声参数，并给出后验均值、不确定性和可视化结果。

**逐行解读**
1. 第 2-3 行设定本次批处理对象：材料类型固定为 `Wool`，孔隙率依次跑 `92` 到 `99`。
2. 第 5-7 行定义 6 个待输出参数名，前 5 个是材料参数，第 6 个 `Sigma Noise` 是统计噪声参数。
3. 第 10 行预分配 `all_rhat_results`，用于保存每个孔隙率下 6 个参数的收敛诊断 `R-hat`。
4. 第 12-15 行开始主循环，每次取一个孔隙率文件夹，并打印当前运行状态。
5. 第 18 行把字符串孔隙率如 `92` 转成数值孔隙率 `phi = 0.92`。
6. 第 21-29 行拼接数据路径，寻找 `R.txt`；如果找不到就跳过该样品。这里路径被硬编码成 `/Users/tao/...`，这是原作者环境相关写法。
7. 第 32-35 行读取原始 `R.txt`，把小数逗号 `,` 改成小数点 `.`，再写成 `converted_R.txt`，目的是让 MATLAB 能正常读表。
8. 第 37-40 行载入清洗后的数据，只保留 100 到 4950 Hz。
9. 第 42-50 行兼容两种列格式：4 列时取第 2 列实部、第 4 列虚部；3 列时取第 2 列实部、第 3 列虚部；同时做 2 点滑动平均降一点噪。
10. 第 52-53 行把实验反射系数组装成复数 `R_exp`，再把实部和虚部上下拼接成一个长向量 `R_exp_concat`，后面的似然函数就是拿它和模型预测做比较。
11. 第 56-66 行调用 [getFluidProperties.m](<./getFluidProperties.m:1>)，从对应 XML 中读样品厚度、温度、气压、湿度，并计算湿空气密度、粘度、声速、Pr 数等；随后把这些量存成 `airProperties` 结构体。
12. 第 68-76 行调用 [getFiberConstraints.m](<./getFiberConstraints.m:1>) 生成 6 个参数的上下界 `lb/ub`，再设置 MCMC 步长、样本数、链数，并在边界内部随机初始化 3 条链。
13. 第 79-82 行调用 [metropolis_hastings.m](<./metropolis_hastings.m:1>) 做 3 条 MCMC 链采样。这个函数内部会反复调用 [log_likelihood.m](<./log_likelihood.m:1>)，而后者又会调用 [jcal_reflection.m](<./jcal_reflection.m:1>) 做 JCAL 正演。
14. 第 84-95 行按参数逐个计算 `R-hat`。含义是比较“链内方差”和“链间方差”，越接近 1 越说明各链混合一致、采样更可信。
15. 第 98-104 行丢弃前 30% burn-in，把 3 条链剩余样本合并，然后计算后验均值 `estimated_params` 和后验标准差 `param_std`。
16. 第 107-124 行把每个孔隙率的结果写入 `*_params.txt`，内容包括参数均值、标准差，以及本次反演使用的上下界。
17. 第 127-150 行从后验样本里随机抽 1000 组，写入 `*_random_params.txt`。这个文件用于后续做不确定性传播或随机预测。
18. 第 152-178 行画两类诊断图：trace plot 看链是否稳定、histogram 看后验分布形状。
19. 第 181-185 行画 `std/mean` 柱状图，用于快速看哪个参数相对不确定性更大。
20. 第 188-206 行用后验均值做一次“最佳估计”正演，比较实验与预测的反射系数和吸声系数。这里还额外调用 [JCAL_O.m](<./JCAL_O.m:1>) 画一条使用固定空气常数的对照曲线。
21. 第 213-238 行把随机抽取的后验样本逐个送入 `jcal_reflection`，得到一簇预测曲线，展示“参数不确定性会导致多大预测散布”。
22. 第 240-243 行是异常保护，某个孔隙率出错不会让整个批处理停掉。
23. 第 247-253 行在所有孔隙率跑完后，统一打印每个样品的 `R-hat`。

**流程图**
```text
开始
  -> 设定材料类型 Wool 和孔隙率列表
  -> 对每个孔隙率循环
     -> 读取 R.txt
     -> 把逗号小数改成点小数
     -> 提取 100-4950 Hz 的复反射系数 R_exp
     -> 读取 XML
     -> 计算湿空气物性和样品厚度
     -> 根据材料类型+孔隙率生成参数边界
     -> 初始化 3 条 MCMC 链
     -> Metropolis-Hastings 采样
        -> 提议新参数
        -> 用 jcal_reflection 计算预测 R
        -> 用 log_likelihood 计算接受概率
        -> 接受或拒绝
     -> 计算 R-hat
     -> 去 burn-in，合并后验样本
     -> 计算后验均值和标准差
     -> 保存参数文件和随机后验样本文件
     -> 画诊断图和后验预测图
  -> 输出所有孔隙率的 R-hat 汇总
结束
```

**6 个参数**
- `Airflow Resistivity`，记作 `sigma`，单位通常是 `Pa·s/m^2`。表示空气穿过多孔骨架时受到的流动阻力。越大通常说明孔道更“堵”，黏滞耗散更强，但太大也会让入射波更难进入材料。
- `Tortuosity`，记作 `alpha_infin`，无量纲。表示孔道路径的曲折程度。越大说明声波在孔隙中看到的等效惯性越大。
- `Viscous Characteristic Length`，记作 `lambda`，单位 `m`。表示黏滞边界层作用的特征孔尺度。越小通常意味着边界层相对更“占空间”，黏滞损耗更明显。
- `Thermal Characteristic Length`，记作 `lambda_prime`，单位 `m`。表示热交换相关的特征孔尺度。它主要影响热边界层和等效体积模量。
- `Static Thermal Permeability`，记作 `k0_prime`，单位 `m^2`。表示热传递/热扩散在静态极限下的“通透性”。它控制 JCAL 模型中热效应那一支的响应。
- `Sigma Noise`，不是材料参数，而是统计模型里的噪声标准差，单位可看作“反射系数误差量级”，本质上是无量纲。它越大，说明模型允许实验与预测存在更大的残差，后验通常也会更宽。

**子文件作用**
- [getFluidProperties.m](<./getFluidProperties.m:1>)：从 XML 提取厚度和环境条件，并计算湿空气物性。
- [getFiberConstraints.m](<./getFiberConstraints.m:1>)：根据纤维直径范围、孔隙率和空气粘度生成参数上下界。
- [metropolis_hastings.m](<./metropolis_hastings.m:1>)：执行 MCMC。
- [log_likelihood.m](<./log_likelihood.m:1>)：定义“实验 R 和模型 R 越接近越好”的高斯似然。
- [jcal_reflection.m](<./jcal_reflection.m:1>)：给定参数后计算反射系数、阻抗和吸声系数。
- [JCAL_O.m](<./JCAL_O.m:1>)：同样是 JCAL 正演，但空气参数用固定常数。

**补充**
- 这个脚本真正体现“不确定性量化”的地方，不是单一最优值，而是 `param_std`、`random_params.txt`、trace/histogram、以及后验预测曲线云图。
- [likelihood.m](<./likelihood.m:1>)、[prior.m](<./prior.m:1>)、[posterior.m](<./posterior.m:1>) 目前并没有被这个主脚本直接调用。
- 如果你后面要真正跑它，优先检查两个硬编码路径：[Uncertainty_quantification.m](<./Uncertainty_quantification.m:21>) 和 [getFluidProperties.m](<./getFluidProperties.m:11>)。

如果你愿意，我下一步可以继续把这份脚本直接改写成“带完整中文注释的副本”，例如生成一个 `Uncertainty_quantification_annotated.m`。

> 我有几个请求和问题：
> 1. 帮我改一下21行的base path，改到我当前目录下
> 2. 实部和虚部取自哪个文件？他们分别代表什么，为什么要分实部和虚部
> 3. 反射系数R_exp为什么是复数？实部和虚部分别代表什么？
> 4. 3条chains是什么意思？
> 5. log_likelihood.m 在算什么？输出是什么？

<details><summary>4 previous messages</summary>

> 我先把硬编码路径改成基于当前 MATLAB 工作目录的写法，并顺手处理 `getFluidProperties.m` 里的 XML 路径，否则主脚本改了以后还是会在读 XML 时卡住。
>
> 我现在直接改文件。改动很小：主脚本的数据根目录改成 `pwd`，XML 读取也同步改成 `pwd`，这样两边路径逻辑一致。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./Uncertainty_quantification.m` (+2 -2)
> - Wrote `./getFluidProperties.m` (+1 -1)
>
> </details>
>
> 路径已经改完。我再核对一下具体行号和内容，然后把你提的 5 个问题按脚本里的上下文解释清楚。
</details>

已经改好了：

- [Uncertainty_quantification.m](<./Uncertainty_quantification.m:21>) 第 21 行改成了 `basePath = pwd;`
- [getFluidProperties.m](<./getFluidProperties.m:11>) 第 11 行也同步改成了 `fullfile(pwd, ...)`

之所以连 `getFluidProperties.m` 也一起改，是因为你只改主脚本的话，后面读 XML 还是会去原来的 `/Users/tao/...` 路径。
注意：现在这套代码依赖 MATLAB 的“当前工作目录”就是项目根目录 `Fibers`。如果你在 MATLAB 里 `cd` 到别的地方，`pwd` 也会跟着变。

1. 实部和虚部取自哪个文件？分别代表什么，为什么要分开？
实部和虚部来自 `R.txt`，脚本先把它转成 `converted_R.txt`，再从这个转换后的文件读取。

对应代码在 [Uncertainty_quantification.m](<./Uncertainty_quantification.m:23>) 到 [Uncertainty_quantification.m](<./Uncertainty_quantification.m:47>)。

以 [Wool/92/R.txt](<./Wool/92/R.txt:1>) 为例，它是 4 列格式：
- 第 1 列：频率
- 第 2 列：反射系数实部
- 第 3 列：频率
- 第 4 列：反射系数虚部

所以代码里是：
- `R_real_filtered = data(:,2)`
- `R_imag_filtered = data(:,4)`

如果文件是 3 列格式，就默认：
- 第 1 列：频率
- 第 2 列：实部
- 第 3 列：虚部

为什么要分实部和虚部：
- 因为反射系数本身是复数，不只是“大小”，还有“相位”
- 只看一个实数，比如只看模长，会丢掉相位信息
- 拟合时同时拟合实部和虚部，等价于同时拟合幅值和相位，信息更完整

2. 反射系数 `R_exp` 为什么是复数？实部和虚部分别代表什么？
`R_exp` 是实验测得的复反射系数，在代码里定义于 [Uncertainty_quantification.m](<./Uncertainty_quantification.m:52>)。

声学里反射系数本质上是：

```math
R = \frac{p_{\text{reflected}}}{p_{\text{incident}}}
```

这里的声压不是普通实数，而是谐波稳态下的复振幅，所以 `R` 也是复数。复数能同时编码两件事：
- 幅值：反射了多少
- 相位：反射波相对入射波滞后或超前多少

实部和虚部的物理意义：
- 实部：与参考相位“同相”的分量
- 虚部：与参考相位相差 90° 的分量

如果写成极坐标形式：

```math
R = |R| e^{j\theta}
```

那么：
- `real(R) = |R| cos(theta)`
- `imag(R) = |R| sin(theta)`

这里最直接有两个用途：
- `|R|^2` 决定能量反射多少，所以代码里有 `alpha = 1 - |R|^2`
- `R` 的相位信息会影响表面阻抗和驻波特性，所以不能只看模长

3. 3 条 chains 是什么意思？
在 [Uncertainty_quantification.m](<./Uncertainty_quantification.m:70>) 到 [Uncertainty_quantification.m](<./Uncertainty_quantification.m:82>)，设置了：

- `n_samples = 1000`
- `n_chains = 3`

这里的 3 条 chains 指的是 3 次独立的 MCMC 采样轨迹。每条链：
- 从不同初值出发
- 按同样的 Metropolis-Hastings 规则往后采样
- 最后比较它们是否收敛到同一个后验分布

为什么要多条链：
- 检查是否收敛，而不是只靠一条链自说自话
- 降低初值影响
- 用 `R-hat` 比较链间差异和链内波动

如果 3 条链最后都混到一起，`R-hat` 会接近 1；如果差很多，说明还没收敛或者卡在不同区域。

4. `log_likelihood.m` 在算什么？输出是什么？
[log_likelihood.m](<./log_likelihood.m:1>) 做的事情是：

给定一组候选参数
`[sigma, alpha_infin, lambda, lambda_prime, k0_prime, sigma_noise]`

去回答这个问题：

“如果这些参数是真的，那么观测到当前实验数据 `R_exp` 的可能性有多大？”

它的流程是：

- 先检查参数是否合法
  - `sigma_noise <= 0` 直接返回 `-Inf`
  - 超出上下界也直接返回 `-Inf`
- 用前 5 个材料参数调用 [jcal_reflection.m](<./jcal_reflection.m:1>)，算出模型预测的复反射系数 `R_predicted`
- 把实验和预测都拆成“实部向量 + 虚部向量”
- 计算残差平方
- 假设残差服从高斯分布，用高斯噪声模型计算对数似然

核心公式在 [log_likelihood.m](<./log_likelihood.m:39>) 到 [log_likelihood.m](<./log_likelihood.m:40>)：

```matlab
log_likelihood_value = -sum(residuals) / (2 * sigma_noise^2) ...
                       - N * log(sigma_noise * sqrt(2 * pi));
```

输出是什么：
- 输出是一个标量 `log_likelihood_value`
- 它不是“概率本身”，而是“概率密度的对数”
- 数值越大，说明这组参数越能解释实验数据
- 数值越小，说明拟合越差
- 如果非法，输出 `-Inf`，表示这组参数直接拒绝

在 MCMC 里，它被用来计算接受率：

```matlab
acceptance_ratio = exp(proposed_log_likelihood - current_log_likelihood);
```

也就是说，新的参数如果让 `log_likelihood` 更大，就更容易被接受。

5. 一个直观总结
这套脚本的核心逻辑其实就是：

- 从 `R.txt` 读实验复反射系数 `R_exp`
- 用 JCAL 模型从参数生成预测反射系数 `R_predicted`
- 用 `log_likelihood` 衡量两者接近程度
- 用 3 条 MCMC 链不断试参数
- 最后得到参数的不确定性分布，而不是一个单点答案

我没有在 MATLAB 里实际跑这份脚本做验证，只完成了路径修改和代码级解释。
如果你要，我下一步可以继续给你做一张“`R.txt -> R_exp -> log_likelihood -> MCMC -> params.txt`”的带公式流程图。
