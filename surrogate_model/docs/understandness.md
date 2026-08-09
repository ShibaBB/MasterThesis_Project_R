# 梳理 surrogate model 代码

> 读一下surrogate model\surrogate_model_building_strategy.md，这是我根据task dexcription做的方案，然后搞清楚surrogate model目录下的所有matlab代码，有调用到其他公式或者文件就去读一下，读完告诉我，我有几个问题不懂需要你讲一下

<details><summary>6 previous messages</summary>

> 我先把 `surrogate model` 目录结构和你提到的 `surrogate_model_building_strategy.md` 读一遍，再顺着 MATLAB 代码里的调用把相关文件补齐。读完我会给你一个清晰的整体理解，之后你直接问不懂的点就行。
>
> 目录名和你消息里的写法不一致，我先在工作区里定位实际路径，再继续把文件逐个读完。
>
> 先把 `surrogate_model` 里的说明和 6 个 MATLAB 脚本完整读掉，然后我会顺着其中的函数调用继续往上游文件追。这样能把“方案 -> 数据生成 -> 检查 -> 训练”这条线先串起来。
>
> 我现在往上游依赖追三层：`generate_surrogate_dataset.m` 的三个核心外部调用，以及它们本身有没有再依赖别的自定义公式或文件。这样读完之后，我对这套代码的“输入从哪来、公式怎么算、数据怎么落盘”会比较完整。
>
> 代码依赖已经基本清楚了。我再补读一层“运行时依赖”：一个代表性的 `Wool_92` XML，以及现有小样本运行生成出来的数据/报告，这样后面你问脚本到底产出了什么、为什么这么设计，我能直接对着实际产物讲。
>
> 我把实际产物也对上了：小数据集确实是 `X: 50x7, Y: 50x64`，检查报告和训练报告也和脚本逻辑一致。最后我再确认一下 `task_description.pdf` 能不能直接抽文本读；能读就补上背景，不能读也不影响我解释代码本身。
</details>

我已经读完了，并且把 `surrogate_model` 目录下 MATLAB 脚本的外部依赖一路追到了实际公式和输入文件。读过的核心内容包括：

- 方案和背景：[surrogate_model_building_strategy.md](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/docs/surrogate_model_building_strategy.md:1>)、[mlp_strategy.md](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/MLP/mlp_strategy.md:1>)、[task_description.pdf](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/docs/task_description.pdf>)
- 数据生成与训练脚本：[generate_teacher_dataset.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/data_generation/generate_teacher_dataset.m:1>)、[inspect_teacher_dataset.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/data_generation/inspect_teacher_dataset.m:1>)、[mlp_train_surrogate_baseline.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/MLP/mlp_train_surrogate_baseline.m:1>)、3 个小样本 driver
- 外部依赖：[getFluidProperties.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/getFluidProperties.m:1>)、[getFiberConstraints.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/getFiberConstraints.m:1>)、[jcal_reflection.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/jcal_reflection.m:1>)
- 代表性运行时文件：[Wool_92.aed1001.xml](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/Wool/92/Wool_92.aed1001.xml>)、小数据检查报告和 baseline 指标报告

我现在对这套代码的理解是这样的：

- 这套 `surrogate_model` 代码不是在做 UQ 主流程，而是在做第一阶段的 surrogate pipeline：把 `jcal_reflection.m` 当成 teacher model，生成 `X -> Y` 数据，再训练一个 MLP 去拟合吸声曲线。
- 当前输入定义是 `X = [phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime]`，输出是 `Y = 100~2000 Hz 上 64 个频点的 alpha(f)`。
- 第一版默认材料是 `Wool`，默认先跑 `porosityfolder = '92'`，但代码结构已经支持 `92~99`。
- `h` 不是手填的，而是 `getFluidProperties(fiberfolder, porosityfolder)` 从 XML 里读 `SampleThickness`，再做 `h = thickness * 1e-3`。对 `Wool/92`，XML 里厚度是 `30 mm`。
- `getFluidProperties.m` 负责从 XML 读 `Thickness / Temperature / AirPressure / RelativeAirHumidity`，然后继续算 humid air 的 `density, eta, gamma, c, kappa, Pr`。
- `getFiberConstraints.m` 根据材料类型的纤维直径范围和 `phi, eta` 给出 6 个上下界；但 surrogate 数据生成只取前 5 个物理 JCAL 参数，最后那个其实是给反演/UQ 用的噪声项，不进入 surrogate 输入。
- `jcal_reflection.m` 里是真正的 JCAL 前向公式链：`rho_omega -> K_omega -> Zc -> k -> Zs -> R -> alpha`，其中 `alpha = 1 - |R|^2`。
- `inspect_teacher_dataset.m` 只做数据体检，不训练；`mlp_train_surrogate_baseline.m` 会做按 porosity 分层切分、对部分输入取 `log10`、标准化，然后训练多输出 MLP。
- 现有小样本产物也对得上：小数据集是 `50 x 7` 输入、`50 x 64` 输出；small baseline 的 test MAE 大约是 `0.0134`，说明流程已经打通，但这还只是 smoke test 规模。

有几个地方很容易让人看晕，我现在已经理清了，后面你可以直接问我：

- 为什么输入是 7 维，为什么 `phi` 和 `h` 要放进来
- `getFiberConstraints.m` 里每个上下界公式到底代表什么
- `jcal_reflection.m` 每一步的物理意义
- 为什么 surrogate 先学 `alpha(f)`，不先学复数反射系数 `R`
- 训练脚本里为什么只对第 `3,5,6,7` 列做 `log10`

你现在直接把不懂的问题一个个抛给我，我可以按“代码含义 + 物理意义 + 这套 pipeline 里它起什么作用”的方式给你讲。

> 回答一下你的第1，4，5个问题

<details><summary>1 previous message</summary>

> 我直接按代码里的当前实现来回答这 3 个点，并把“为什么这样设计”和“如果以后要改，会改哪里”一起讲清楚。
</details>

1. 输入是 7 维，是因为当前 surrogate 不是在学“某个固定样品的一条曲线”，而是在学前向模型里这 7 个显式变量到吸声曲线的映射。代码里已经把输入顺序写死为 `[phi, h, sigma, alpha_infinity, lambda, lambda_prime, k0_prime]`，见 [generate_teacher_dataset.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/data_generation/generate_teacher_dataset.m:72>) 和 [generate_teacher_dataset.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/data_generation/generate_teacher_dataset.m:117>)。`phi` 和 `h` 要放进去，不是因为它们现在变化很大，而是因为它们本来就是 [jcal_reflection.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/jcal_reflection.m:1>) 的直接输入，物理上会影响 `rho_omega`、`K_omega`、`Zs` 和最后的 `alpha`。尤其 `h` 直接出现在 `coth(j*k*h)` 里，见 [jcal_reflection.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/jcal_reflection.m:34>)；`phi` 则进了动态密度和动态体积模量公式，见 [jcal_reflection.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/jcal_reflection.m:19>) 和 [jcal_reflection.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/jcal_reflection.m:23>)。
当前这份 `Wool + 92` 的小数据集里，`phi` 和 `h` 实际上是常数，所以这批数据“有效自由维度”只有 5 个；但接口保留 7 维，是为了以后自然扩展到 `92-99` 和不同厚度，而不用重写模型结构。这也正是 [mlp_strategy.md](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/MLP/mlp_strategy.md:105>) 的设计意图。

4. 先学 `alpha(f)`，不先学复反射系数 `R`，核心原因是“目标更直接、训练更简单、也更符合你当前 thesis 的 QoI”。代码里 `alpha` 本来就是由 `R` 派生出来的，`alpha = 1 - |R|^2`，见 [jcal_reflection.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/jcal_reflection.m:46>)。如果你学 `R`，网络就得处理复数输出，实际要拆成 `Re(R)` 和 `Im(R)` 两路，或者幅值和相位两路；这会立刻引入相位包裹、正负号、复数一致性这些额外难点。相反，`alpha` 是实数、通常在 `[0,1]` 附近、图像也更容易看，和任务描述里“100–2000 Hz 的吸声表现”直接一致。
代价当然也有：学 `alpha` 会丢掉相位信息，所以它不适合以后做所有和反射相位、阻抗实虚部直接相关的任务。但对你现在这个“先验证 surrogate 能不能贴住 teacher model”的第一阶段，这是更稳的选择。[mlp_strategy.md](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/MLP/mlp_strategy.md:127>) 到 [mlp_strategy.md](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/MLP/mlp_strategy.md:139>) 其实就是这个判断。

5. 训练脚本里只对第 `3,5,6,7` 列做 `log10`，因为这 4 列正好是 `sigma, lambda, lambda_prime, k0_prime`，见 [mlp_train_surrogate_baseline.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/MLP/mlp_train_surrogate_baseline.m:25>) 和 [generate_teacher_dataset.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/data_generation/generate_teacher_dataset.m:72>)。这几类量的共同点是：都严格为正，而且量级跨度明显，尤其 `k0_prime` 很容易跨好几个数量级，`sigma`、`lambda`、`lambda_prime` 也常常不是线性均匀分布。对它们先取 `log10`，相当于把“乘法尺度”压成“加法尺度”，会让网络更容易学，也避免某几个大数值特征在训练里支配梯度。
而 `phi`、`h`、`alpha_infinity` 没进 `log10`，是因为它们通常范围窄、量级稳定、而且接近 `O(1)` 或固定厚度。比如现在 `phi=0.92`，`h=0.03 m`，`alpha_infinity` 也基本在 1 附近到 2 左右，直接标准化通常就够了。注意脚本是“先 `log10`，再标准化”，不是二选一，见 [mlp_train_surrogate_baseline.m](</C:/Users/liuzi/OneDrive/Master Thesis/Fibers/surrogate_model/MLP/mlp_train_surrogate_baseline.m:156>)。
如果以后你把 `h` 也放宽到很大的正范围，或者引入更多工况，`log10_feature_indices` 是可以改的，不是物理定律，只是当前这版 preprocessing 的经验性选择。

如果你愿意，下一步我可以继续讲两个最容易卡住的点：

1. `getFiberConstraints.m` 里 5 个参数上下界公式各自是什么意思。
2. `jcal_reflection.m` 从 `rho_omega` 到 `alpha` 的每一步物理意义。
