# surrogate model 初步搭建

> 读一下我surrogate model文件夹下的surrogate_model_building_strategy.md，顺便读一下其中提到的需要调用的文件，先不开始搭模型，解读一下这个strategy的内容，分析一下接下来要怎么一步一步操作

<details><summary>10 previous messages</summary>

> 我先把 `surrogate_model/surrogate_model_building_strategy.md` 和它直接提到的相关文件读一遍，梳理这份 strategy 的目标、依赖和执行顺序，再给你一个按步骤推进的操作方案。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `Get-Content -Raw 'surrogate_model/surrogate_model_building_strategy.md'`
> - Ran `rg -n "\.m|\.md|\.pdf|MATLAB|script|function|file|task_description|UQ|likelihood|surrogate" 'surrogate_model/surrogate_model_building_strategy.md'`
>
> </details>
>
> 我已经把 strategy 主体读完了。下一步我会把它明确依赖的 `jcal_reflection.m`、`getFiberConstraints.m`、`Uncertainty_quantification.m`、`log_likelihood.m` 和现有参数样本文件过一遍，确认这份方案和你现有代码是否一致。
>
> <details><summary>Ran 5 commands</summary>
>
> - Ran `Get-Content -Raw 'getFiberConstraints.m'`
> - Ran `Get-Content -Raw 'jcal_reflection.m'`
> - Ran `Get-Content -Raw 'Uncertainty_quantification.m'`
> - Ran `Get-Content -Raw 'log_likelihood.m'`
> - Ran `Get-ChildItem -Recurse -File | Where-Object { $_.Name -match '(_params\.txt|_random_params\.txt)$' } | Select-Object -ExpandProperty FullName`
>
> </details>
>
> 现在可以看出 strategy 的判断基本是基于你现有的反演/UQ 工作流写的，不是空泛方案。我再抽样看几个 `*_params.txt` / `*_random_params.txt`，确认这些文件里到底能提供什么信息给后续 surrogate。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-Content -TotalCount 40 'Wool_95_params.txt'`
> - Ran `Get-Content -TotalCount 20 'Wool_95_random_params.txt'`
> - Ran `Get-Content -TotalCount 80 'Extracted_All_Params.txt'`
> - Ran `Get-ChildItem 'surrogate_model' -File | Select-Object -ExpandProperty Name`
>
> </details>
>
> `surrogate_model` 目录里只有 strategy 和一个 `task_description.pdf`。我会再快速看一下你现有的任务清单文件，避免 strategy 和你之前列的执行项之间有冲突。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `Get-Content -Raw 'surrogate_model/porous_absorber_surrogate_model_task_list.md'`
> - Ran `Get-Item 'surrogate_model/task_description.pdf' | Select-Object Name,Length,LastWriteTime`
>
> </details>
>
> 任务清单文件不在 `surrogate_model/` 子目录里，我先在工作区里定位它；然后我就给你一个基于现有代码的分步解释，不会开始建模。
>
> <details><summary>Ran 5 commands</summary>
>
> - Ran `rg --files | rg 'porous_absorber_surrogate_model_task_list\.md|task_description\.pdf'`
> - Ran `rg -n "getFluidProperties|metropolis_hastings|JCAL_O" 'Uncertainty_quantification.m'`
> - Ran `rg -n "function log_likelihood|getFiberConstraints|sigma_noise|jcal_reflection|R_predicted_concat|log_likelihood_value" 'log_likelihood.m'`
> - Ran `rg -n "function \[Reflect|Dynamic density|alpha = 1 - abs\(Reflect\)\.\^2|Zs = Zc .* coth|Check for valid input" 'jcal_reflection.m'`
> - Ran `rg -n "function \[lb, ub\]|if strcmp\(fiberfolder|lb =|ub =" 'getFiberConstraints.m'`
>
> </details>
</details>

我读完了这份 strategy，也对它依赖的核心文件做了交叉检查。结论先说：这份方案的方向是对的，而且和你现有代码是能接上的。它本质上不是“直接把现在的 MATLAB 模型当 surrogate”，而是把现有 JCAL MATLAB 模型当成 `teacher / forward model`，先批量造数据，再训练一个真正的 surrogate。

**Strategy 在说什么**

[surrogate_model_building_strategy.md](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/surrogate_model_building_strategy.md:1) 的核心逻辑很清楚：

1. 现有 `jcal_reflection.m` 是前向物理模型，不是 surrogate。
2. 用这个前向模型生成大量 `参数 -> 吸声曲线` 数据。
3. 再用 DNN 去拟合这个映射。
4. 训练好的 DNN 才用于后续 UQ、敏感性分析、可靠性分析。

这个判断和你现有代码是一致的，因为 [jcal_reflection.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/jcal_reflection.m:1) 已经能从 `h, phi, sigma, alpha_infin, lambda, lambda_prime, k0_prime, freq, airProperties` 计算反射系数和吸声系数，最终 `alpha = 1 - abs(Reflect).^2` 在第 47 行。

**和现有文件的关系**

现有代码里，最关键的几个文件分别承担这些角色：

- [jcal_reflection.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/jcal_reflection.m:1)
  - 真正可复用的前向求解器。后面生成 surrogate 训练集时，最核心就是反复调用它。
- [getFiberConstraints.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/getFiberConstraints.m:1)
  - 给定 `fiber type + phi + airProperties`，返回 5 个 JCAL 参数和 1 个噪声参数的上下界。这里实际上已经把“物理合理范围”编码进去了。
- [Uncertainty_quantification.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Uncertainty_quantification.m:1)
  - 这是你现有的反演/UQ 主流程。它做的事是：读取实验 `R`，固定 `fiberfolder` 和 `phi`，从 XML/实验条件拿到 `h` 和空气参数，然后 MCMC 反推出材料参数，再保存 `*_params.txt` 和 `*_random_params.txt`。
- [log_likelihood.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/log_likelihood.m:1)
  - 这是反演用的，不是 surrogate 训练核心。它用 `jcal_reflection` 预测复反射系数，再和实验 `R` 做高斯似然比较。

这也解释了 strategy 为什么说 `log_likelihood.m` 和 `metropolis_hastings.m` 不是第一阶段重点：因为你现在要做的是前向 surrogate，不是继续做参数反演。

**我看到的几个关键点**

1. 你现有反演结果文件确实能直接服务 surrogate 的参数范围定义。
   比如 [Extracted_All_Params.txt](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Extracted_All_Params.txt:1) 已经汇总了不同纤维、不同孔隙率下的后验均值和标准差；像 `Wool_95_random_params.txt` 这种文件则给了后验样本。

2. `*_random_params.txt` 里包含 6 列参数，但第 6 列是 `Sigma Noise`，这不是 surrogate 的材料输入。
   也就是说，后面如果你拿这些随机样本做 surrogate 训练候选输入，只能取前 5 列材料参数；`sigma_noise` 只属于反演统计模型，不属于吸声前向物理映射。

3. strategy 里建议的 surrogate 输入是
   `[phi, sigma, alpha_infinity, lambda, lambda_prime, k0_prime, h]`，
   但你现有 UQ 流程里 `phi` 和 `h` 目前更像“样品条件/元数据”，不是和其他参数同等地随机采样的变量。
   这个是你接下来最需要先定清楚的设计决策之一。

4. 另一个必须先定的点是 `fiber type`。
   因为 [getFiberConstraints.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/getFiberConstraints.m:1) 明确按 `Acrylic / Silk / Wool` 切不同约束，而且三类材料在 [Extracted_All_Params.txt](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Extracted_All_Params.txt:1) 里的参数量级差异很大。
   所以第一版最稳妥的做法通常不是“一次把三类纤维全混在一起建一个 surrogate”，而是：
   - 先只做一个 fiber type
   - 甚至先只做一个较窄的 `phi` 范围
   - 把流程跑通后再扩展

**接下来应该怎么一步一步做**

如果先不开始建模，我建议你下一阶段按这个顺序推进：

1. 先定“第一版 surrogate 的作用域”
   最重要的是回答三个问题：
   - 是只做一个 fiber type，还是三种一起做
   - `phi` 是固定还是作为输入变量
   - `h` 是固定还是作为输入变量
   从你现有代码结构看，最稳的 MVP 是：`单一 fiber type + task-specific phi 范围 + h 作为已知输入或固定值`。

2. 统一 surrogate 的输入定义
   你现在实际候选输入有两层：
   - 材料参数层：`sigma, alpha_infinity, lambda, lambda_prime, k0_prime`
   - 样品/工况层：`phi, h`，以及可能隐含的 `fiber type`
   这一步要明确哪些是真正给网络的输入，哪些只是用来限定采样范围的条件。

3. 统一输出定义和频率网格
   strategy 建议直接学 `alpha(f)`，这是合理的，因为它比学复反射系数 `R` 更简单，也更接近最终工程量。
   你接下来只需要定：
   - 频率范围，比如 100–2000 Hz
   - 频率点数，比如 64、100 或 128 个固定点
   一旦频率网格固定，输出维度就固定了。

4. 先做参数范围整理，不急着采样
   这一步应该综合三类信息：
   - [getFiberConstraints.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/getFiberConstraints.m:1) 的物理边界
   - [Extracted_All_Params.txt](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Extracted_All_Params.txt:1) 的后验均值/标准差
   - 各个 `*_random_params.txt` 的后验样本分布
   目标不是立刻出最终范围，而是先定一个“第一版 task-specific training domain”。

5. 再决定训练样本从哪里来
   这里有两条路：
   - 纯物理边界内做 LHS 采样
   - 以后验样本为中心扩展采样
   对你现在这个项目，第二种更适合先做第一版，因为更贴近现有样品分布，训练难度也更低。

6. 最后才是写数据生成脚本
   真正开始动手时，你第一段代码不该是 DNN，而该是一个很干净的 dataset builder：
   - 输入一批参数
   - 调 `jcal_reflection`
   - 输出统一频率网格上的 `alpha`
   - 保存成 `X, Y`

**我对这份 strategy 的评价**

这份 strategy 最大的优点是路线很实际：它把 thesis 的主线从“被 COMSOL 或大型数值仿真卡住”转成了“先把一个可控、可验证、可批量采样的 surrogate pipeline 跑通”。这对你现在的代码基础是匹配的。

它现在唯一还不够落地的地方，不在建模方法，而在“问题定义还差最后一层具体化”：

- 第一版到底只做哪类 fiber
- `phi`/`h` 到底是输入变量还是条件
- 训练范围到底按多宽来设
- 输出频率网格到底怎么定

这些一旦定下来，后面的工作流其实就很顺了。

如果你愿意，我下一步可以不写代码，只继续帮你做两件事中的一件：

1. 把这份 strategy 进一步收敛成“你这个项目的第一版 surrogate 具体定义”
2. 基于你现有 `*_params.txt` 和 `*_random_params.txt`，帮你先设计一版合理的参数范围方案

> 你可以读一下Uncertainty_quantification.m这个脚本，参考一下这里的h是怎么取的，就可以按照同样的方法取; phi可以先取92，但要像jcal_reflection脚本一样，具备计算从92到99的能力。样品也模仿jcal脚本，可以通过定义的方法更换材料，不要混着材料一起做，第一版先从wood开始。我现在可以先一步一步来，你就假设我的jcal_reflection是真实物理的，那就先不用UQ来验证他，就当他是可靠的老师，先做surrogate来贴近MATLAB。频率范围和网格就先按你建议的，从100--2000HZ，频率点数先用64，这两个变量定义的时候方便后面可以更改。

<details><summary>2 previous messages</summary>

> 我按你刚补充的约束，把现有 `Uncertainty_quantification.m` 里 `h` 的取法和材料切换方式再对齐一遍，然后给你一版已经收敛过的第一版 surrogate 实施顺序。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `rg -n "function .*getFluidProperties|getFluidProperties\(" -S .`
> - Ran `Get-Content -Raw 'Uncertainty_quantification.m'`
> - Ran `rg -n "fiberfolder =|porosity_list|phi =|h = thickness|getFluidProperties|jcal_reflection\(" 'Uncertainty_quantification.m'`
>
> </details>
</details>

可以，方向现在已经很明确了。我按 [Uncertainty_quantification.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Uncertainty_quantification.m:1) 里的现有做法，把第一版 surrogate 的问题定义收敛成下面这样。

先说两点和你现有脚本完全对齐的约定：

- `h` 的取法就照 `Uncertainty_quantification.m` 来。也就是先通过 `getFluidProperties(fiberfolder, porosityfolder)` 读取样品厚度 `thickness`，然后用 `h = thickness * 1e-3` 转成米。对应位置在 [Uncertainty_quantification.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Uncertainty_quantification.m:78) 和 [Uncertainty_quantification.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Uncertainty_quantification.m:81)。
- `phi` 的管理方式也照这个脚本来。也就是外部先定义 `porosityfolder = '92' ... '99'`，再用 `phi = str2double(porosityfolder) / 100` 转成数值。对应位置在 [Uncertainty_quantification.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Uncertainty_quantification.m:18) 和 [Uncertainty_quantification.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Uncertainty_quantification.m:25)。

另外，你说的 `wood` 我这里按你现有工程里的 `Wool` 理解，因为当前材料名在脚本里就是 `fiberfolder = 'Wool'`，位置在 [Uncertainty_quantification.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Uncertainty_quantification.m:4)。如果你真的是想新加 `Wood`，那是另一件事；如果只是口误，那第一版就按 `Wool` 做。

**第一版 surrogate 的明确设定**

现在先固定成这个版本，不掺 UQ，不混材料：

- 材料只做 `Wool`
- 默认起点孔隙率先用 `92`
- 但脚本结构要能切换 `92` 到 `99`
- `h` 不手填，统一从 `getFluidProperties` 读取
- 把 `jcal_reflection.m` 当成可靠老师模型
- surrogate 目标只做 `MATLAB JCAL -> DNN 拟合`
- 频率范围固定为 `100` 到 `2000 Hz`
- 频率点数先固定为 `64`
- 这两个量做成独立配置变量，后面可改

在这个设定下，第一版最自然的输入输出定义是：

- 输入：`[phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime]`
- 输出：64 个固定频率点上的 `alpha(f)`

这样做的理由很直接：虽然第一轮你可以先从 `phi = 0.92` 起步，但脚本本身以后要支持 `92-99`，那 `phi` 最终就该作为可切换变量纳入整体设计；而 `h` 在你的现有工作流里本来就是随样品读取的，也应该作为输入的一部分保留下来。

**接下来一步一步怎么做**

我建议严格按下面顺序推进，不要一上来写 DNN。

1. 先定一个专门的 surrogate 配置层
   这里先只定义变量，不做训练：
   - `fiberfolder = 'Wool'`
   - `porosity_list = {'92','93',...,'99'}`
   - `selected_porosity = '92'` 作为第一轮默认值
   - `freq_min = 100`
   - `freq_max = 2000`
   - `n_freq = 64`

2. 写清楚“老师模型调用规范”
   这一步的目标不是训练，而是明确以后 dataset builder 每次怎么拿一条样本：
   - 根据 `fiberfolder` 和 `porosityfolder` 调 `getFluidProperties`
   - 取 `thickness`
   - 计算 `h = thickness * 1e-3`
   - 计算 `phi = str2double(porosityfolder)/100`
   - 生成固定频率网格 `freq_grid = linspace(100, 2000, 64)`
   - 把参数送进 `jcal_reflection`

3. 先定第一版训练域
   因为你现在不想先引入 UQ 验证，那第一步就不要碰 `*_random_params.txt` 去做后验传播。
   第一版只需要确定：在 `Wool + phi=0.92` 下，5 个 JCAL 参数各自在什么范围内采样。
   这一步范围来源依然可以参考：
   - `getFiberConstraints.m`
   - `Wool_92_params.txt`
   - `Wool_92_random_params.txt`
   但用途只是“定训练域”，不是“做 UQ”。

4. 然后写数据生成脚本，不写训练脚本
   第一份真正该写的代码应该是 dataset generator，而不是神经网络。
   它做三件事：
   - 采样一组输入参数
   - 调 `jcal_reflection`
   - 存成 `X, Y`
   到这一步为止，你甚至还没开始 surrogate training，但整个工程最关键的地基已经稳了。

5. 数据生成跑通后，再做 baseline DNN
   这时再进入：
   - 训练集/验证集/测试集切分
   - 输入标准化
   - 可选 log 变换
   - 小型 MLP 拟合 64 维 `alpha(f)`

6. 最后才做 surrogate 验证
   因为你已经明确“先假设 `jcal_reflection` 是真实老师”，所以这里的验证目标就很单纯：
   - DNN 是否逼近 MATLAB
   - 看 test MSE / MAE
   - 看几条随机吸声曲线对比
   - 看 worst-case curve

**你现在这个阶段最合适的下一步**

不是训练 DNN，而是先把“数据生成规范”定死。也就是把下面这几个决定固化下来：

- 材料固定 `Wool`
- 默认 porosity 从 `92` 起
- 但保留 `92-99` 的切换能力
- `h` 统一从 `getFluidProperties` 取
- 频率网格统一用 `linspace(100, 2000, 64)`
- 输出统一为 `alpha`

如果你愿意，下一步我可以直接开始做这个第一步：先帮你写一个专门的 MATLAB 数据生成脚本骨架，只负责生成 surrogate 训练数据，不碰 DNN 训练。

> 先把我们聊到目前为止的策略优化一下之前的strategy, 写一个strategy1.md，

<details><summary>2 previous messages</summary>

> 我先把我们刚才收敛下来的约束整理成一份更落地的 `strategy1.md`，重点是把第一版 surrogate 的范围、变量定义和执行顺序写清楚，替换掉原先比较泛的部分。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./surrogate_model/strategy1.md` (+334 -0)
>
> </details>
</details>

已经写好了新版策略文件：[strategy1.md](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/strategy1.md:1)。

这版相对原来的 `surrogate_model_building_strategy.md` 做了几处关键收敛：

- 明确第一版只做 `Wool`，不混材料
- 明确 `h` 按 [Uncertainty_quantification.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/Uncertainty_quantification.m:78) 里的 `getFluidProperties -> thickness * 1e-3` 来取
- 明确从 `porosityfolder='92'` 起步，但脚本结构保留 `92-99` 扩展能力
- 明确第一版只把 `jcal_reflection.m` 当老师模型，不先引入 UQ 验证
- 明确频率网格先固定为 `100-2000 Hz`、`64` 点，并要求写成可配置变量
- 明确下一步优先写“数据生成脚本”，而不是先写 DNN

如果你愿意，下一步我可以直接接着把这个 strategy 落成一个 MATLAB 数据生成脚本骨架。

> 可以，在surrogate model文件夹下写，如果要注释的话就用英文

<details><summary>6 previous messages</summary>

> 我现在开始在 `surrogate_model/` 下落一个 MATLAB 数据生成脚本骨架，先对齐你现有工程里 `getFluidProperties`、`airProperties` 和 `jcal_reflection` 的调用方式，再把可配置变量和数据输出结构写进去。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-ChildItem -Name`
> - Ran `rg -n "lhsdesign|randperm|readmatrix|save\(|writematrix|matfile" -S .`
> - Ran `rg -n "function .*getFluidProperties|getFluidProperties\(" -S .`
>
> </details>
>
> 我已经确认了 `getFluidProperties.m` 在工程根目录可直接复用，`surrogate_model/` 里目前还没有数据生成脚本。接下来我会新建一个独立的 MATLAB 脚本，默认走 `Wool + 92`，同时保留 `92-99` 的切换能力，并把频率网格、样本数和保存路径都做成配置项。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./surrogate_model/generate_surrogate_dataset.m` (+155 -0)
>
> </details>
>
> 脚本已经写进 `surrogate_model/` 了。我再快速检查一下关键配置和输出字段，确认它和我们刚才定的策略一致。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `rg -n "fiberfolder =|available_porosityfolders|selected_porosityfolders|freq_min|freq_max|n_freq|n_samples|sampling_method|output_file|X =|Y =|getFluidProperties|getFiberConstraints|jcal_reflection|save\(" 'surrogate_model/generate_surrogate_dataset.m'`
> - Ran `Get-Content -TotalCount 260 'surrogate_model/generate_surrogate_dataset.m'`
>
> </details>
</details>

已经在 [generate_surrogate_dataset.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/generate_surrogate_dataset.m:1) 写好了第一版数据生成脚本，注释也都用了英文。

这版脚本已经按我们刚才定的策略落地了：默认 `fiberfolder = 'Wool'`、`selected_porosityfolders = {'92'}`，但保留了 `92-99` 的切换能力；`h` 按 `getFluidProperties -> thickness * 1e-3` 取；频率网格由 `freq_min = 100`、`freq_max = 2000`、`n_freq = 64` 控制；输出数据保存为 `X` 和 `Y`，其中 `X` 是 7 维输入，`Y` 是 64 点吸声曲线。

目前还没实际跑 MATLAB 验证语法或生成数据。如果你下一步要继续，我建议直接先做一轮很小的试跑，比如把 `n_samples` 临时改成 `20` 或 `50`，先确认 `getFluidProperties`、`getFiberConstraints` 和 `jcal_reflection` 都能顺利串起来。之后我可以继续帮你补下一步，例如加数据可视化检查，或者直接写训练脚本骨架。

> 可以，继续。写出训练骨架脚本并配上一些可视化内容，按照工业级别去写，如果有特别不确定的地方先来找我确定再执行

<details><summary>12 previous messages</summary>

> 我继续往下落训练骨架，但先按“可执行工程脚手架”来写，不会擅自引入太多方法分支。先看一下仓库里有没有现成的 MATLAB 深度学习用法可复用；如果没有，我会默认按 MATLAB Deep Learning Toolbox 的标准写法实现，并把依赖写清楚。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-ChildItem -Name 'surrogate_model'`
> - Ran `rg -n "trainNetwork|dlnetwork|featureInputLayer|regressionLayer|trainingOptions|deepNetworkDesigner" -S .`
> - Ran `Get-Content -Raw 'surrogate_model/generate_surrogate_dataset.m'`
>
> </details>
>
> 训练脚本我会按“可运行骨架 + 可追踪产物”的方式写：包含数据读取、分层切分、预处理、MLP 训练、测试集评估和图像导出。默认我会用 MATLAB Deep Learning Toolbox 的标准 `trainNetwork` 流程；如果你这里没有这个工具箱，脚本会直接报清楚依赖，不会静默退化。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./surrogate_model/train_surrogate_baseline.m` (+535 -0)
>
> </details>
>
> 训练骨架已经写进去了，我先做一次静态检查，主要看配置项、产物路径、评估逻辑和图像导出是不是完整闭合。这里先不运行 MATLAB，只做代码层面的收口。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `Get-Content -TotalCount 420 'surrogate_model/train_surrogate_baseline.m'`
> - Ran `rg -n "dataset_file|artifacts_dir|split_ratios|apply_log10_to_inputs|hidden_layer_sizes|max_epochs|trainNetwork|predict\(|metrics_report|exportgraphics|function " 'surrogate_model/train_surrogate_baseline.m'`
>
> </details>
>
> 我发现脚本里有两处用了较新的 `any(...,'all')` 写法。为避免 MATLAB 版本差异，我把它改成更兼容的形式，这样老一点的版本也不容易卡在预处理上。
>
> <details><summary>Edited a file, ran 4 commands</summary>
>
> - Wrote `./surrogate_model/train_surrogate_baseline.m` (+0 -0)
> - Wrote `./surrogate_model/train_surrogate_baseline.m` (+0 -0)
> - Ran `rg -n "any\(X_work\(:, .*all\)" 'surrogate_model/train_surrogate_baseline.m'`
> - Ran `Get-Content 'surrogate_model/train_surrogate_baseline.m' | Select-Object -Index 248..280`
> - Ran `Get-Content 'surrogate_model/train_surrogate_baseline.m' | Select-Object -Skip 248 -First 40`
> - Ran `Get-Content 'surrogate_model/train_surrogate_baseline.m' | Select-Object -Skip 270 -First 20`
> - Wrote `./surrogate_model/train_surrogate_baseline.m` (+2 -2)
>
> </details>
>
> 我顺手把那两处兼容性判断再收了一下，避免使用 MATLAB 对临时表达式索引支持不一致的写法。这个地方宁可啰嗦一点，也不要留下版本兼容隐患。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./surrogate_model/train_surrogate_baseline.m` (+4 -2)
>
> </details>
>
> 我再补一个稳妥性改动：图像导出不再硬依赖 `exportgraphics`。如果你的 MATLAB 版本没有这个函数，脚本会自动退回 `saveas`，这样训练本身不会因为画图环节中断。
>
> <details><summary>Edited a file, ran 2 commands</summary>
>
> - Wrote `./surrogate_model/train_surrogate_baseline.m` (+14 -6)
> - Ran `Get-Content 'surrogate_model/train_surrogate_baseline.m' | Select-Object -Skip 240 -First 60`
> - Ran `Get-Content 'surrogate_model/train_surrogate_baseline.m' | Select-Object -Skip 395 -First 170`
>
> </details>
</details>

已经继续写好了训练骨架脚本：[train_surrogate_baseline.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/train_surrogate_baseline.m:1)。

这版是按“工程可复用脚手架”写的，包含这些部分：

- 读取 [generate_surrogate_dataset.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/generate_surrogate_dataset.m:1) 生成的 `.mat` 数据集
- 按 porosity 做分层切分，得到 train / validation / test
- 对输入做可配置 `log10` 变换和标准化
- 对输出做标准化后训练，再自动反变换回原始 `alpha`
- 用 MLP baseline 做 64 维多输出回归
- 保存模型、预测结果、预处理参数和文本指标报告
- 自动导出可视化图

目前会导出的可视化包括：

- training history
- predicted vs true scatter
- error vs frequency
- random test curve comparisons
- worst-case curve comparisons
- test curve MAE histogram

产物默认会落到：
- `surrogate_model/artifacts/wool_baseline_mlp/`

我还顺手把兼容性收了一下：
- 如果没有 Deep Learning Toolbox，脚本会直接明确报错
- 如果 MATLAB 版本没有 `exportgraphics`，会自动退回 `saveas`

我还没有实际替你运行训练。如果你要继续，我建议下一步先做这两个动作中的一个：

1. 先把 `generate_surrogate_dataset.m` 的 `n_samples` 临时设成 `50` 或 `100`，跑一轮小数据，再跑训练脚本做通路验证
2. 我直接继续帮你补一个“训练前数据检查脚本”，专门看 `X/Y` 分布、参数范围、曲线形状是否合理

有一个我认为值得你拍板但还不至于阻塞当前脚手架的问题：第一版 baseline 你是否就固定使用 MATLAB Deep Learning Toolbox。如果你想改成 Python / PyTorch 训练、MATLAB 只负责造数据，这会影响后续目录结构和模型保存方式。

> MATLAB在功能上会与python 有差距吗？如果不引入python效果会变差？

会有差距，但对你现在这个阶段，不引入 Python 不会天然让效果变差。

更准确地说，差距主要在“生态和灵活性”，不在“你能不能先做出一个可靠 surrogate”。

MATLAB 的优点是：

- 你现在的老师模型本来就在 MATLAB 里，数据生成和训练放在同一环境最顺
- 对 MLP 这种标准回归网络，MATLAB Deep Learning Toolbox 足够做 baseline
- 工程链更短，不需要处理 MATLAB/Python 数据交接、环境管理、依赖冲突

Python 的优势主要在后期：

- 深度学习生态更丰富，PyTorch / TensorFlow 更灵活
- 更容易做复杂模型、定制损失、超参数搜索、GPU 训练控制
- 后面如果要接更先进的 surrogate 方法、主动学习、贝叶斯神经网络，Python 选择更多

对你这个项目当前阶段，关键问题不是“哪个框架理论上更强”，而是：

- 先把 `JCAL -> dataset -> baseline surrogate -> validation` 这条链跑通
- 证明 surrogate 能稳定逼近 MATLAB

这个目标上，MATLAB 完全够用。第一版效果的决定因素主要是：

- 数据集质量
- 参数采样范围是否合理
- 频率网格定义是否稳定
- 预处理是否正确
- 网络大小和训练配置是否合适

这些比“MATLAB 还是 Python”重要得多。

所以我的建议很明确：

- 第一版先全部留在 MATLAB
- 等 baseline 跑通后，如果你发现需要更复杂的模型或训练控制，再迁到 Python
- 不要在 surrogate 还没建立前就引入双语言工作流，那只会增加复杂度

如果你愿意，我下一步可以继续帮你做一个 MATLAB 的“训练前数据检查脚本”，这样你先能判断生成的数据是否适合训练。

> 可以，先做一个这个脚本，跑一轮小数据

<details><summary>28 previous messages</summary>

> 我先补一个训练前数据检查脚本，然后尝试用小样本把 `generate_surrogate_dataset.m` 跑一轮，确认数据链条是通的。如果本机 MATLAB 命令不可用，我会停在可执行脚本层并把阻塞点明确给你。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -Raw 'getFluidProperties.m'`
> - Ran `Get-Command matlab -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source`
> - Ran `Get-Command octave -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source`
>
> </details>
>
> 我先把数据生成脚本稍微工程化一下，让它支持外部配置覆盖，这样我们可以不手改默认参数就跑小样本测试。然后我会补一个数据检查脚本，并直接用 MATLAB `-batch` 跑一轮小数据。
>
> <details><summary>Created a file, edited a file</summary>
>
> - Wrote `./surrogate_model/generate_surrogate_dataset.m` (+12 -0)
> - Wrote `./surrogate_model/inspect_surrogate_dataset.m` (+276 -0)
>
> </details>
>
> 数据检查脚本已经补好了。现在我直接用 MATLAB 跑一轮 `Wool + 92 + 50 samples` 的小数据，并把检查脚本也跑起来，看看这条链有没有实质性错误。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `matlab -batch "$surrogate_dataset_config = struct(''n_samples'', 50, ''selected_porosityfolders'', {{''92''}}, ''output_file'', fullfile(pwd, ''surrogate_model'', ''generated_data'', ''Wool_surrogate_dataset_small.mat'')); run(''surrogate_model/generate_surrogate_dataset.m'');"`
> - Ran `matlab -batch "$surrogate_inspection_config = struct(''dataset_file'', fullfile(pwd, ''surrogate_model'', ''generated_data'', ''Wool_surrogate_dataset_small.mat''), ''inspection_name'', ''wool_dataset_inspection_small''); run(''surrogate_model/inspect_surrogate_dataset.m'');"`
>
> </details>
>
> 命令行转义出了问题，不是脚本本身报错。我改用 PowerShell 更稳的引号方式重新跑一次，避免 `$surrogate_*` 被 shell 提前展开。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `matlab -batch 'surrogate_dataset_config = struct("n_samples", 50, "selected_porosityfolders", {{"92"}}, "output_file", fullfile(pwd, "surrogate_model", "generated_data", "Wool_surrogate_dataset_small.mat")); run("surrogate_model/generate_surrogate_dataset.m");'`
> - Ran `matlab -batch 'surrogate_inspection_config = struct("dataset_file", fullfile(pwd, "surrogate_model", "generated_data", "Wool_surrogate_dataset_small.mat"), "inspection_name", "wool_dataset_inspection_small"); run("surrogate_model/inspect_surrogate_dataset.m");'`
>
> </details>
>
> 我不继续跟命令行转义较劲了，直接落两个小型 driver 脚本来跑测试，这样更稳，也更适合后面重复使用。顺便我会把配置覆盖逻辑再做得更直接一点。
>
> <details><summary>Created 2 files, edited 2 files</summary>
>
> - Wrote `./surrogate_model/generate_surrogate_dataset.m` (+11 -9)
> - Wrote `./surrogate_model/inspect_surrogate_dataset.m` (+6 -9)
> - Wrote `./surrogate_model/run_small_dataset_generation.m` (+12 -0)
> - Wrote `./surrogate_model/run_small_dataset_inspection.m` (+11 -0)
>
> </details>
>
> driver 脚本已经加好了。现在我直接用它们跑一次真实的 MATLAB 小样本流程，优先确认数据生成和检查两段都能走通。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `matlab -batch "run('surrogate_model/run_small_dataset_generation.m')"`
> - Ran `matlab -batch "run('surrogate_model/run_small_dataset_inspection.m')"`
>
> </details>
>
> MATLAB 的 `run` 会带来当前目录变化，这正好暴露出路径处理还不够稳。我把这些脚本统一改成基于脚本自身位置推导项目根目录，这样后面你从哪里启动都不会乱。
>
> <details><summary>Edited 5 files</summary>
>
> - Wrote `./surrogate_model/generate_surrogate_dataset.m` (+4 -1)
> - Wrote `./surrogate_model/inspect_surrogate_dataset.m` (+5 -2)
> - Wrote `./surrogate_model/train_surrogate_baseline.m` (+5 -2)
> - Wrote `./surrogate_model/run_small_dataset_generation.m` (+5 -2)
> - Wrote `./surrogate_model/run_small_dataset_inspection.m` (+5 -2)
>
> </details>
>
> 还有一个隐含路径点：`getFluidProperties.m` 里面自己用的是 `pwd`。为了不改你的原始物理脚本，我在数据生成脚本里临时切到项目根目录再执行，这样就能和你现有工程保持一致。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./surrogate_model/generate_surrogate_dataset.m` (+3 -0)
>
> </details>
>
> 路径问题已经收住了。我现在重新跑一次小样本生成和检查，看看 MATLAB 端是否已经完全打通。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `matlab -batch "run('surrogate_model/run_small_dataset_generation.m')"`
> - Ran `matlab -batch "run('surrogate_model/run_small_dataset_inspection.m')"`
>
> </details>
>
> 我看到一个真实问题了：脚本开头的 `clear` 把 wrapper 传进去的配置覆盖掉了，所以小样本参数没有生效。这个地方要改成“保留配置变量”。另外数据生成和检查也要顺序跑，不能并行。
>
> <details><summary>Edited 3 files</summary>
>
> - Wrote `./surrogate_model/generate_surrogate_dataset.m` (+1 -1)
> - Wrote `./surrogate_model/inspect_surrogate_dataset.m` (+1 -1)
> - Wrote `./surrogate_model/train_surrogate_baseline.m` (+1 -1)
>
> </details>
>
> 配置传递的问题已经修掉了。我现在顺序跑：先生成 `50` 个小样本数据，再在同一个小数据文件上跑检查脚本。
>
> <details><summary>Ran a command</summary>
>
> - Ran `matlab -batch "run('surrogate_model/run_small_dataset_generation.m')"`
>
> </details>
>
> 小数据已经生成成功了，当前唯一的非致命提示是这台 MATLAB 没有 `lhsdesign`，所以这轮自动退回了均匀随机采样。现在继续跑检查脚本，看数据产物和可视化是否正常输出。
>
> <details><summary>Ran a command</summary>
>
> - Ran `matlab -batch "run('surrogate_model/run_small_dataset_inspection.m')"`
>
> </details>
>
> 检查脚本已经跑通了，不过我发现一个小的工程细节：`inspection_name` 覆盖后目录名没有跟着刷新。这个不影响结果内容，但我把它修一下，避免后面多次实验时产物目录混淆。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./surrogate_model/inspect_surrogate_dataset.m` (+18 -2)
>
> </details>
>
> 我把目录名覆盖逻辑修好了，再补跑一次小数据检查，这样产物就会落到你期望的 `wool_dataset_inspection_small` 目录里。
>
> <details><summary>Ran a command</summary>
>
> - Ran `matlab -batch "run('surrogate_model/run_small_dataset_inspection.m')"`
>
> </details>
</details>

已经把训练前数据检查脚本写好，并实际跑通了一轮小数据。

新增的脚本有：

- [inspect_surrogate_dataset.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/inspect_surrogate_dataset.m:1)
- [run_small_dataset_generation.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/run_small_dataset_generation.m:1)
- [run_small_dataset_inspection.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/run_small_dataset_inspection.m:1)

这次我实际跑了两步：

1. 生成 `Wool + porosity 92 + 50 samples` 的小数据
   输出文件在 [Wool_surrogate_dataset_small.mat](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/generated_data/Wool_surrogate_dataset_small.mat)

2. 对这份小数据做检查和可视化
   产物目录在 [wool_dataset_inspection_small](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_dataset_inspection_small)

检查脚本会输出：

- 文本报告 [dataset_inspection_report.txt](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_dataset_inspection_small/dataset_inspection_report.txt)
- MATLAB 汇总文件 [dataset_inspection_summary.mat](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_dataset_inspection_small/dataset_inspection_summary.mat)
- 一组图，包含特征直方图、吸声带宽图、随机曲线、相关矩阵、porosity 计数、曲线极值分布

这轮运行里唯一值得注意的点是：

- 你的 MATLAB 当前环境没有 `lhsdesign`，所以数据生成自动退回到了均匀随机采样
- 这不影响小样本通路验证，但如果你坚持第一版就用严格的 Latin Hypercube Sampling，我建议我下一步给你补一个不依赖 toolbox 的自定义 LHS 实现

现在这条链已经通了：`XML/metadata -> jcal_reflection -> dataset -> inspection artifacts`。

下一步我建议直接做一件事：
- 用这份小数据继续试跑 [train_surrogate_baseline.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/train_surrogate_baseline.m:1)

如果你同意，我下一步就不再停留分析，直接给训练脚本也加一个 small-run driver，并替你跑一轮 baseline。

> 我现在在MATLAB中安装好了lhs，所以严格保留lhs，你帮我优化一下并且试一下

<details><summary>6 previous messages</summary>

> 我先把数据生成脚本改成“`lhs` 就必须严格用 `lhsdesign`，不可静默退回随机采样”，然后重新跑一轮小数据，确认你当前 MATLAB 环境里的 LHS 已经真正生效。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./surrogate_model/generate_surrogate_dataset.m` (+2 -2)
>
> </details>
>
> 我已经把采样逻辑收紧了。现在如果你选 `lhs`，脚本只会使用真正的 Latin Hypercube Sampling；环境不满足时会直接报错，不再偷偷退回随机采样。下面我直接重跑小样本生成。
>
> <details><summary>Ran a command</summary>
>
> - Ran `matlab -batch "run('surrogate_model/run_small_dataset_generation.m')"`
>
> </details>
>
> 这次没有再出现回退 warning，说明 `lhsdesign` 已经被实际调用了。我再顺手把小数据检查脚本跑一遍，确认新生成的数据产物仍然正常。
>
> <details><summary>Ran a command</summary>
>
> - Ran `matlab -batch "run('surrogate_model/run_small_dataset_inspection.m')"`
>
> </details>
</details>

已经按你的要求优化好了，并且实测通过。

我做了一个关键改动：[generate_surrogate_dataset.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/generate_surrogate_dataset.m:1) 现在对 `sampling_method = 'lhs'` 是严格模式：

- 如果 `lhsdesign` 可用，就用真正的 LHS
- 如果不可用，直接报错
- 不再静默退回 `random`

然后我重新实际跑了一轮小样本：

- 运行脚本：[run_small_dataset_generation.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/run_small_dataset_generation.m:1)
- 成功生成：`50` 个 `Wool + 92` 样本
- 输出文件：[Wool_surrogate_dataset_small.mat](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/generated_data/Wool_surrogate_dataset_small.mat)

这次运行里没有再出现之前那个
`lhsdesign not found. Falling back to uniform random sampling.`
的 warning，所以可以确认现在已经是严格 LHS 采样了。

我也把检查脚本重新跑了一遍：

- 运行脚本：[run_small_dataset_inspection.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/run_small_dataset_inspection.m:1)
- 检查产物目录：[wool_dataset_inspection_small](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_dataset_inspection_small)

当前状态是稳定的：
- 数据生成通
- 数据检查通
- LHS 严格保留

下一步最自然的是继续试跑训练骨架。我可以直接给 [train_surrogate_baseline.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/train_surrogate_baseline.m:1) 也加一个 small-run driver，然后用这份 `50` 样本数据先跑一轮 baseline 训练。

> 可以

<details><summary>8 previous messages</summary>

> 我继续把训练脚本接上 small-run driver，并用刚才那份 `50` 样本数据实际跑一轮 baseline。先把训练脚本的配置覆盖能力补齐，再执行。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./surrogate_model/train_surrogate_baseline.m` (+41 -0)
>
> </details>
>
> 训练脚本现在支持外部配置覆盖了。我接着加一个小样本训练 driver，默认直接吃 `Wool_surrogate_dataset_small.mat`，并把 epoch 和 batch size 调到适合 smoke test 的级别。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./surrogate_model/run_small_baseline_training.m` (+22 -0)
>
> </details>
>
> driver 已经就位。我现在实际跑一轮小样本 baseline 训练，看看 Deep Learning Toolbox、数据切分和评估图导出能不能全部打通。
>
> <details><summary>Ran a command</summary>
>
> - Ran `matlab -batch "run('surrogate_model/run_small_baseline_training.m')"`
>
> </details>
>
> 训练在工具链层面卡住了，不是脚本逻辑问题。我先快速检查一下你当前 MATLAB 里到底装了哪些相关工具箱，这样我能给你一个准确的替代方案，而不是猜。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `matlab -batch "disp(exist('trainNetwork','file')); disp(exist('fitrnet','file')); disp(exist('fitnet','file')); ver"`
> - Ran `Get-ChildItem 'surrogate_model/artifacts' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name`
>
> </details>
</details>

现在的真实情况很明确：

- `trainNetwork` 不可用
- `fitnet` 也不可用
- `fitrnet` 可用
- 你当前安装的是 `Statistics and Machine Learning Toolbox`，没有 `Deep Learning Toolbox`

所以我写的 DNN/MLP 训练骨架目前不能直接跑，不是代码坏了，而是 MATLAB 功能集不匹配。

这里有两个可行方向：

1. 我把训练脚本改成基于 `fitrnet` 的 MATLAB 原生神经网络回归方案
   这条路你现在就能继续跑，不用装新工具箱。
   代价是它不是 `trainNetwork` 风格的深度学习脚本，灵活性会低一些，但做第一版 surrogate baseline 完全够用。

2. 你安装 `Deep Learning Toolbox`，我保留当前 MLP 训练骨架
   这条路更接近我们原本的 DNN 设计，但现在会被工具箱阻塞。

我的建议是选 `1`，直接把现有训练骨架切到 `fitrnet`，先把 baseline 跑通。
如果你同意，我下一步就直接把 [train_surrogate_baseline.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/train_surrogate_baseline.m:1) 改成 `fitrnet` 版本，并重新跑一轮小数据训练。

> 我现在装好了deep learning toolbox，所以保留当前的MLP训练骨架继续

<details><summary>4 previous messages</summary>

> 我先确认 `Deep Learning Toolbox` 现在在 MATLAB 里已经可见，然后直接重跑小样本 baseline，继续沿用当前的 MLP 训练骨架。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `matlab -batch "disp(exist('trainNetwork','file')); disp(exist('featureInputLayer','file')); ver"`
> - Ran `Get-Content -TotalCount 140 'surrogate_model/run_small_baseline_training.m'`
>
> </details>
>
> 工具箱已经就位了。我现在直接跑 small-run 训练，优先看训练、预测、指标和图像导出这一整条链能不能一次打通。
>
> <details><summary>Ran a command</summary>
>
> - Ran `matlab -batch "run('surrogate_model/run_small_baseline_training.m')"`
>
> </details>
</details>

small-run baseline 已经实际跑通了，当前的 MLP 训练骨架可以继续保留。

这次我做了两件事：

- 给 [train_surrogate_baseline.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/train_surrogate_baseline.m:1) 补了外部配置覆盖能力
- 新增了 small-run driver：[run_small_baseline_training.m](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/run_small_baseline_training.m:1)

我实际跑的是：

- 数据集：[Wool_surrogate_dataset_small.mat](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/generated_data/Wool_surrogate_dataset_small.mat)
- 训练产物目录：[wool_baseline_mlp_small](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_baseline_mlp_small)

这轮 small-run 的结果是：

- Train MAE: `0.008986`
- Validation MAE: `0.015167`
- Test MAE: `0.013355`

说明当前这条链已经通了：

- `JCAL teacher model`
- `dataset generation`
- `inspection`
- `MLP training`
- `prediction + metrics + figures export`

训练产物里会有这些关键文件：

- 模型：[surrogate_baseline_model.mat](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_baseline_mlp_small/surrogate_baseline_model.mat)
- 预测结果：[surrogate_baseline_predictions.mat](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_baseline_mlp_small/surrogate_baseline_predictions.mat)
- 指标报告：[metrics_report.txt](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_baseline_mlp_small/metrics_report.txt)

下一步最合理的不是继续改代码结构，而是把样本量从 `50` 提到一个更像样的 baseline 规模，比如 `500` 或 `1000`，再看误差和曲线拟合是否稳定。
如果你同意，我下一步可以直接帮你做一个“baseline experiment driver”，用 `1000` 个 LHS 样本正式跑第一版 baseline。

> 如果我自己去matlab里面跑要怎么运行？输出要怎么看怎么分析？

你现在可以直接在 MATLAB 里按这条顺序跑。

**怎么运行**

先切到项目根目录：
```matlab
cd('C:\Users\liuzi\OneDrive\Master Thesis\Fibers')
```

然后按阶段运行。

1. 生成小数据
```matlab
run('surrogate_model/run_small_dataset_generation.m')
```

2. 检查小数据
```matlab
run('surrogate_model/run_small_dataset_inspection.m')
```

3. 训练小样本 baseline
```matlab
run('surrogate_model/run_small_baseline_training.m')
```

如果你要跑正式一点的数据，不用 small driver，而是直接运行主脚本前先给配置：

```matlab
surrogate_dataset_config = struct();
surrogate_dataset_config.n_samples = 1000;
surrogate_dataset_config.selected_porosityfolders = {'92'};
surrogate_dataset_config.output_file = fullfile(pwd, 'surrogate_model', 'generated_data', 'Wool_surrogate_dataset_1000.mat');
run('surrogate_model/generate_surrogate_dataset.m')
```

然后检查：
```matlab
surrogate_inspection_config = struct();
surrogate_inspection_config.dataset_file = fullfile(pwd, 'surrogate_model', 'generated_data', 'Wool_surrogate_dataset_1000.mat');
surrogate_inspection_config.inspection_name = 'wool_dataset_inspection_1000';
run('surrogate_model/inspect_surrogate_dataset.m')
```

再训练：
```matlab
surrogate_training_config = struct();
surrogate_training_config.dataset_file = fullfile(pwd, 'surrogate_model', 'generated_data', 'Wool_surrogate_dataset_1000.mat');
surrogate_training_config.experiment_name = 'wool_baseline_mlp_1000';
run('surrogate_model/train_surrogate_baseline.m')
```

**输出在哪里看**

数据生成后看这里：

- 数据文件：`surrogate_model/generated_data/`
- 例如 [Wool_surrogate_dataset_small.mat](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/generated_data/Wool_surrogate_dataset_small.mat)

检查结果看这里：

- `surrogate_model/artifacts/wool_dataset_inspection_small/`
- 重点看 [dataset_inspection_report.txt](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_dataset_inspection_small/dataset_inspection_report.txt)
- 再看 `figures/` 里的图

训练结果看这里：

- `surrogate_model/artifacts/wool_baseline_mlp_small/`
- 重点看 [metrics_report.txt](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_baseline_mlp_small/metrics_report.txt)
- 再看 `figures/` 里的图
- 模型和预测结果在：
  - [surrogate_baseline_model.mat](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_baseline_mlp_small/surrogate_baseline_model.mat)
  - [surrogate_baseline_predictions.mat](/C:/Users/liuzi/OneDrive/Master%20Thesis/Fibers/surrogate_model/artifacts/wool_baseline_mlp_small/surrogate_baseline_predictions.mat)

**怎么分析数据检查结果**

先看 `dataset_inspection_report.txt`：

- `Feature Summary`
  - 看每个输入参数的 `min/max/mean/std` 是否合理
  - 如果范围异常窄，网络学不到泛化
  - 如果范围异常宽，第一版会更难学
- `Alpha Summary`
  - 看 `alpha` 是否基本落在合理范围
  - 如果经常超出 `[0,1]` 太多，要回头查老师模型输出和采样范围

再看图：

- `feature_histograms.png`
  - 看输入分布是否均匀、是否有明显空洞
- `alpha_band.png`
  - 看吸声曲线整体形状是否合理，5%-95% band 是否过宽或过窄
- `random_absorption_curves.png`
  - 直接看曲线是不是像正常的吸声曲线
- `feature_correlation_matrix.png`
  - 看输入之间是否有很强相关性
- `curve_extrema_histograms.png`
  - 看每条曲线的最小值/最大值分布是否异常

**怎么分析训练结果**

先看 `metrics_report.txt`：

- `Train MAE`
  - 训练集拟合误差
- `Validation MAE`
  - 调参时最重要
- `Test MAE`
  - 最终泛化能力
- `R2`
  - 越接近 `1` 越好

判断原则：

- `train` 很低、`validation/test` 明显高
  - 过拟合
- `train/validation/test` 都高
  - 欠拟合，可能样本太少、网络太小、参数范围太宽
- 三者接近且都低
  - 当前 baseline 是健康的

再看图：

- `training_history.png`
  - 看训练损失和验证损失是否一起下降
  - 如果训练降很多、验证不降，就是过拟合
- `predicted_vs_true_scatter.png`
  - 点越贴近对角线越好
- `error_vs_frequency.png`
  - 看误差是不是集中在某些频率段
  - 如果高频或峰值附近误差特别大，说明这部分更难学
- `random_curve_comparisons.png`
  - 看随机样本曲线拟合是否形状一致
- `worst_case_curves.png`
  - 这是最重要的诊断图之一
  - 看最差样本到底是整体偏移，还是峰位没学准
- `curve_error_histogram.png`
  - 看大多数样本是否都低误差，还是只有少数样本很差

**你可以在 MATLAB 里直接读结果**

比如：
```matlab
load('surrogate_model/artifacts/wool_baseline_mlp_small/surrogate_baseline_predictions.mat')
```

然后自己看某一条测试曲线：
```matlab
idx = 1;
plot(freq_grid, Y_test_raw(idx,:), 'k-', 'LineWidth', 1.5); hold on
plot(freq_grid, Y_test_pred(idx,:), 'r--', 'LineWidth', 1.5);
legend('True','Predicted')
grid on
```

看整体误差：
```matlab
mean(abs(Y_test_pred - Y_test_raw), 'all')
```

看每条曲线误差：
```matlab
curve_mae = mean(abs(Y_test_pred - Y_test_raw), 2);
histogram(curve_mae)
```

如果你愿意，我下一步可以直接给你写一个 `how_to_run_and_read_results.md` 放进 `surrogate_model/`，把这些运行命令和结果解读整理成一份你后面随时可查的说明。
