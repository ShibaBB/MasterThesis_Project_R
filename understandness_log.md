# 阅读 working_history.dm

> 读一下working_history.dm文件，这是我之前工作时候的对话，里面详细介绍了主程序uncertainty_quantification.m在算什么以及调用了哪些公式和文件，读一下里面提到的文件便于更方便理解，你理解了告诉我一声

<details><summary>6 previous messages</summary>

> 我先读 `working_history.md`，同时核对其中提到的核心脚本和数据文件，整理出 `Uncertainty_quantification.m` 的计算流程和依赖关系。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `Get-Content -Path 'working_history.md'`
> - Ran `Get-ChildItem -Name`
>
> </details>
>
> `working_history.md` 里是之前的分析记录，但终端默认编码把中文读乱了。我改用明确编码，再把里面提到的几个核心 `.m` 文件和样例数据一起过一遍。
>
> <details><summary>Ran 8 commands</summary>
>
> - Ran `Get-Content -Path 'getFiberConstraints.m' -Encoding utf8`
> - Ran `Get-Content -Path 'log_likelihood.m' -Encoding utf8`
> - Ran `Get-Content -Path 'Uncertainty_quantification.m' -Encoding utf8`
> - Ran `Get-Content -Path 'jcal_reflection.m' -Encoding utf8`
> - Ran `Get-Content -Path 'working_history.md' -Encoding utf8`
> - Ran `Get-Content -Path 'metropolis_hastings.m' -Encoding utf8`
> - Ran `Get-Content -Path 'getFluidProperties.m' -Encoding utf8`
> - Ran `Get-Content -Path 'JCAL_O.m' -Encoding utf8`
>
> </details>
>
> 主程序和 6 个核心函数我已经读完了。再补一眼样例输入 `Wool/92/R.txt` 和对应 XML，确认实验数据列结构和环境参数来源。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `Get-Content -Path 'Wool/92/R.txt' -Encoding utf8 -TotalCount 12`
> - Ran `Get-Content -Path 'Wool/92/Wool_92.aed1001.xml' -Encoding utf8 -TotalCount 80`
>
> </details>
</details>

我读完了，已经理解这套代码在做什么。

核心结论是：`[Uncertainty_quantification.m](/abs/path?)` wait need absolute paths. Need provide proper links maybe not necessary? We should provide clickable file links with absolute targets. Let's craft with absolute path. Need not exceed.

> 好的，在流程图中提到了用metropolis-hastings采样，下面用log_likelihood 计算接受概率，这个接受概率大概是一个什么范围的值，有没有好坏的评判标准，下面的接受和拒绝又是什么，表示这次拟合的结果是否采纳？

这里的“接受概率”不是一个固定应该落在某个数值区间的质量分数，它是 MCMC 里每一步用来决定“要不要把新参数收下”的概率。

在这份代码里，[metropolis_hastings.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/metropolis_hastings.m:23>) 先算

```matlab
acceptance_ratio = exp(proposed_log_likelihood - current_log_likelihood);
```

然后用

```matlab
if rand() < acceptance_ratio
```

决定接受还是拒绝。

含义可以直接这样理解：

- 如果新参数比当前参数拟合更好，`proposed_log_likelihood > current_log_likelihood`，那么指数项大于 1，这时代码里一定会接受。
- 如果新参数更差，接受率就在 `0` 到 `1` 之间，差得越多，值越接近 `0`。
- 所以严格说，这里的“接受概率”实际使用时可以看成 `min(1, acceptance_ratio)`，虽然代码里没显式写这个 `min`，但 `rand()` 只在 `0` 到 `1` 之间，所以效果等价。

它的大致范围是：

- 最小接近 `0`
- 最大可大于 `1`，但这等价于“100% 接受”

没有“单步接受概率越大越好”这种简单标准。判断好坏要看整体接受率，不是某一步。

常见经验是：

- 接受率太低，比如长期接近 `0%` 到 `10%`：说明提议步子太大，链总在提很差的点，几乎走不动。
- 接受率太高，比如长期 `80%` 到 `100%`：说明步子太小，虽然老是接受，但探索得很慢，样本相关性会很强。
- 比较常见的可用区间大概是 `20%` 到 `50%`。
- 对随机游走型 Metropolis，很多人会把大约 `20%` 到 `30%` 左右看成比较合理，但这不是硬标准。

“接受”和“拒绝”是什么意思：

- 接受：这一步提出的新参数被采纳，链移动到新位置。
- 拒绝：这一步新参数不采纳，链留在旧位置，当前样本会重复记一次旧参数。

这不表示“这次拟合成功/失败”，也不表示“最终结果是否有效”。它只表示：

- 这一步采样时，新参数有没有进入后验样本序列。

真正最终用于结果的是整条链累计出来的大量样本，再经过 burn-in 后合并求均值、标准差。也就是说：

- 接受/拒绝是 MCMC 采样过程中的局部决策
- 不是对最终拟合结果的一次性裁决

如果你愿意，我下一步可以直接把这段流程补进 `working_history.md`，写成一段更适合你后面复习的中文说明。

> 你可以帮我把log_likelihood.m中逐条解释以下吗，直接注释在文件中

<details><summary>2 previous messages</summary>

> 我先直接在 `log_likelihood.m` 里加逐段中文注释，保留原有逻辑不改，只增强可读性。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./log_likelihood.m` (+19 -1)
>
> </details>
</details>

我已经把逐段中文注释直接加进了 [log_likelihood.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/log_likelihood.m:1>)，逻辑没改，只补充了每一步在做什么，重点包括：

- 参数边界检查为什么会直接返回 `-Inf`
- 第 6 个参数 `sigma_noise` 的统计意义
- 为什么要把复反射系数拆成实部和虚部再拼接
- 残差平方和与高斯对数似然公式各自代表什么
- 这个函数返回的标量为什么能用于 MCMC 的接受判断

如果你要，我下一步可以继续把 [metropolis_hastings.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/metropolis_hastings.m:1>) 也用同样方式逐条注释掉，这样你就能把“提议参数 -> 算 log likelihood -> 接受/拒绝”整条链一次看清。

> 于是总的对数似然就是：
>     % 1. 残差平方和项：残差越小，对数似然越大
>     % 2. 归一化项：噪声 sigma_noise 越小，对拟合误差的容忍度越低
> 怎么理解这个归一化项，sigma_noise， 他不是判定参数是否被采纳的值吗？为什么他的大小能代表拟合误差的容忍度

`sigma_noise` 不是“是否采纳”的那个值。它是你这个模型里的一个参数，用来描述“实验数据和模型预测之间允许有多大的随机误差”。

真正决定采不采纳的是 `acceptance_ratio`，也就是在 [metropolis_hastings.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/metropolis_hastings.m:23>) 里算的

```matlab
acceptance_ratio = exp(proposed_log_likelihood - current_log_likelihood);
```

`sigma_noise` 只是参与了 `log_likelihood` 的计算，而 `log_likelihood` 又进一步影响接受率。关系是：

`params` 里包含 `sigma_noise` -> 算出 `log_likelihood` -> 再算 `acceptance_ratio` -> 决定接受或拒绝

你问的“归一化项”是这个：

```matlab
- N * log(sigma_noise * sqrt(2 * pi))
```

它来自高斯分布的标准形式。因为这里默认每个残差都满足

```math
e_i \sim \mathcal{N}(0,\sigma_{noise}^2)
```

所以单个残差的概率密度是

```math
p(e_i|\sigma)=\frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{e_i^2}{2\sigma^2}\right)
```

取对数以后就会分成两部分：

```math
\log p = -\frac{e_i^2}{2\sigma^2} - \log(\sigma\sqrt{2\pi})
```

对所有数据点求和，就得到你现在代码里的形式。

怎么理解“容忍度”：

1. 第一项 `-sum(residuals)/(2*sigma_noise^2)` 控制“残差惩罚有多重”。
   `sigma_noise` 越小，分母越小，同样大小的残差会被罚得更狠。
   这就是“模型更不容忍误差”。

2. 第二项 `-N*log(sigma_noise*sqrt(2*pi))` 是高斯分布必须带上的归一化常数。
   它不是额外拍脑袋加的惩罚，而是概率密度本身的一部分。
   它的作用是防止模型总想把 `sigma_noise` 取到无限大来逃避残差惩罚。

所以两项是配合的：

- `sigma_noise` 大：
  残差项惩罚变轻，说明“允许更大误差”
  但归一化项会变得更负，说明“你不能白白把噪声设很大”

- `sigma_noise` 小：
  归一化项会更有利
  但残差项会罚得很重，只要拟合稍微不好，`log_likelihood` 就会很差

因此 `sigma_noise` 本质上是在平衡一句话：

“这些实验数据和模型之间的偏差，到底是因为参数不对，还是因为观测噪声本来就这么大？”

你可以把它理解成“误差条的宽度”：

- 误差条很宽，说明模型觉得数据有较大噪声，大一点的残差也还能接受
- 误差条很窄，说明模型要求预测必须非常贴近实验，否则似然就会很低

所以“`sigma_noise` 越大，容忍度越高”说的是它在高斯噪声模型中的统计意义，不是说它自己直接决定采纳。直接决定采纳的是由整个 `log_likelihood` 算出来的接受率。

如果你愿意，我可以继续把这一段也直接补成 `log_likelihood.m` 里的中文注释，专门解释“为什么归一化项不能省略”。

> metropolis_haslings的输出是什么，标量还是矩阵？log_likelihood脚本的输入是什么，是一条MCMC chain中猜出的所有Params，还是只有一组params，换句话说，跑一次log_likelihood会用到所有params的值吗？我问这个问题的原因是在log_likelihood中比较了新参数的采纳概率，如果不是很多参数，怎么会有比较这个过程？

`metropolis_hastings.m` 的输出是一个矩阵，不是标量。

在 [metropolis_hastings.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/metropolis_hastings.m:7>) 里：

```matlab
samples = zeros(n_samples, length(init_params));
```

所以它输出的 `samples` 维度是：

- 行数：`n_samples`
- 列数：参数个数，这里是 6

也就是每一行是一轮 MCMC 迭代后保存下来的那一组参数。例如第 `i` 行就是第 `i` 次采样时链所在位置的 6 个参数值。

而在主程序 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:77>) 里，又把 3 条链都存进了：

```matlab
all_samples(chain, :, :)
```

所以最终 `all_samples` 是一个三维数组：

- 第 1 维：第几条 chain
- 第 2 维：第几次采样
- 第 3 维：第几个参数

`log_likelihood.m` 的输入里，`params` 只是一组参数，不是整条链，也不是所有历史参数。

也就是说，跑一次 [log_likelihood.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/log_likelihood.m:1>) 时，用到的是当前某一个候选点，比如：

```matlab
params = [sigma, alpha_infin, lambda, lambda_prime, k0_prime, sigma_noise]
```

它会用这 6 个数：

1. 调 `jcal_reflection`
2. 算出这一组参数对应的预测反射系数
3. 和实验 `R_exp` 比较
4. 返回一个标量 `log_likelihood_value`

所以：

- `metropolis_hastings` 输出：矩阵
- `log_likelihood` 输出：标量
- `log_likelihood` 每次输入：只有一组 `params`
- 这一组 `params` 内部包含全部 6 个参数值，但不是整条 chain 的所有样本

你疑惑的关键点在这里：`log_likelihood` 里并没有直接比较“新参数的采纳概率”。

它做的只是给“某一组参数”打一个分，这个分就是对数似然。真正的比较发生在 `metropolis_hastings.m` 里。

过程是：

1. 当前链在一个位置 `current_params`
2. 生成一个新提议 `proposed_params`
3. 分别计算两次：
   - `current_log_likelihood = log_likelihood(current_params, ...)`
   - `proposed_log_likelihood = log_likelihood(proposed_params, ...)`
4. 用这两个标量比较，得到接受率

也就是：

```matlab
acceptance_ratio = exp(proposed_log_likelihood - current_log_likelihood);
```

所以“比较”不是在 `log_likelihood.m` 里面做的，而是：

- `log_likelihood` 负责给单个参数点打分
- `metropolis_hastings` 负责拿“新点分数”和“旧点分数”作比较

可以把它理解成爬山：

- `params` 是山上的一个点
- `log_likelihood(params)` 是这个点的海拔
- `metropolis_hastings` 每次提一个新点
- 然后比较“新点海拔”和“旧点海拔”
- 更高通常接受，更低有时也接受一点，这样才不会太容易卡死在局部区域

所以并不是一次 `log_likelihood` 就完成“采纳判断”，而是需要至少两个参数点：

- 当前点 `current_params`
- 新提议点 `proposed_params`

各自跑一次 `log_likelihood`，再在 `metropolis_hastings` 里比较。

如果你要，我下一步可以直接把 [metropolis_hastings.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/metropolis_hastings.m:1>) 也逐条加中文注释，并专门把“current/proposed/log-likelihood/acceptance_ratio”这几步标清楚。

> 把uncertainty_quantification 和metropolis_hastings也逐行中文备注一下

<details><summary>2 previous messages</summary>

> 我直接给这两个脚本补逐段中文注释，保留现有逻辑和变量名不变，只增强你后面自己读代码时的可解释性。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./Uncertainty_quantification.m` (+64 -0)
> - Wrote `./metropolis_hastings.m` (+28 -0)
>
> </details>
</details>

我已经把中文注释补进了 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:1>) 和 [metropolis_hastings.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/metropolis_hastings.m:1>)，没有改算法，只补了解释。

现在这两个文件里重点已经标清了：主程序的数据流、3 条链和 `all_samples` 的维度、`R-hat`/burn-in/后验均值的含义，以及 `metropolis_hastings` 里“当前点 -> 提议点 -> 算新旧 log likelihood -> 接受或拒绝”的过程。

如果你继续往下读，下一步最适合补注释的是 [jcal_reflection.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/jcal_reflection.m:1>)，因为那部分是整个物理模型最难读的一段。

> 因为metropolis_hastings的采样是根据概率来决定要不要新参数，因此是不是三条chain上的n_samples数量都不一样？

不是。在你这份代码里，3 条 chain 的 `n_samples` 是一样的，都会严格等于你设定的 `1000`。

原因在 [metropolis_hastings.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/metropolis_hastings.m:7>)：

```matlab
samples = zeros(n_samples, length(init_params));
```

然后循环是固定跑到：

```matlab
for i = 2:n_samples
```

所以每条链无论接受了多少次、拒绝了多少次，都会保存 `n_samples` 行样本。

关键点是：MCMC 里的“拒绝”不等于“这一步没有样本”。

如果某一步提议的新参数没被接受，链不会少记一个样本，而是：

- 保持在旧参数位置不动
- 再把这个旧位置重复记录一次到 `samples(i,:)`

所以：

- 接受时：第 `i` 个样本是新参数
- 拒绝时：第 `i` 个样本和第 `i-1` 个样本相同

这就是为什么 3 条链的样本数相同，但“有效信息量”可能不同：

- 如果某条链拒绝很多次，它虽然也有 1000 个样本
- 但其中会有很多重复值
- 说明链移动慢、相关性强、有效样本数偏少

所以你可以区分两个概念：

- `n_samples`
  这是“记录下来的样本个数”，在三条链里是一样的
- 有效样本数
  这是“真正有多少独立信息量”，不同链可能不一样

简单说，概率决定的是“链这一步动不动”，不是“这一条链有没有这一行样本”。

如果你愿意，我下一步可以再给你画一个很短的例子，比如一条 chain 连续 6 步里哪些步接受、哪些步拒绝，最后 `samples` 矩阵会长什么样。

> 我其实还没有太懂链这个东西的作用是什么，我用MCMC分布不就是想要得到一个范围内所有参数的分布，然后看看那个参数的概率分布吗？往前移动的意思是在运算+stepsize时还有不成功的时候？如果一步的stepsize没加上我不就调用不到ub边界值进行运算了？而且为什么说有效样本，原地不动的样本没有意义吗？

你理解的大方向是对的：你用 MCMC 的目的，确实是想得到参数的后验分布，而不是只要一个最优点。
但“链”这个东西，是实现这个目标的具体方法。它不是一次把整个参数空间都算完，而是用“随机游走”的方式，在高概率区域里逐步采样。

先把几个概念拆开。

**1. 链是什么**

一条 chain 就是一串按顺序产生的参数点：

```text
params(1) -> params(2) -> params(3) -> ...
```

每个 `params(i)` 都是 6 个参数的一组取值。
所以链不是“某一个参数的曲线”，而是“参数空间里一个点接一个点走出来的轨迹”。

MCMC 不是在整个区间里把所有可能参数都枚举一遍，那样维度一高就根本算不动。
它做的是：

- 先随便找一个起点
- 每次提一个附近的新点
- 如果新点更合理，就倾向接受
- 如果差一点，也有小概率接受
- 这样走很久以后，链在什么区域停留得多，那个区域的后验概率就高

所以最后你看到的“参数分布”，其实是由这条链长期停留的位置统计出来的。

**2. 为什么要“往前移动”**

“往前移动”不是数轴上加一个步长那么简单，而是参数空间里的位置更新。

在你的代码里：

```matlab
proposed_params = current_params + step_size .* randn(size(current_params));
```

意思是：

- 对 6 个参数同时加一个随机扰动
- 扰动有正有负
- 大小由 `step_size` 控制

所以它不是“固定每次 +step_size”，而是“围绕当前点随机提议一个附近的新点”。

“移动成功”表示：

- 这个新提议点被接受
- 链的位置从 `current_params` 变成 `proposed_params`

“移动失败”表示：

- 虽然提了个新点，但它没有被接受
- 链还留在原地

所以你说的“还有不成功的时候”是对的，但这里“不成功”不是步长没加上，而是：

- 新点算出来了
- 但比较之后觉得不值得去
- 所以这一步不换位置

**3. 为什么不一定会走到 `ub` 边界**

MCMC 的目的不是把 `lb` 到 `ub` 每个地方都走到一遍，更不是专门去碰边界。
`lb` 和 `ub` 只是允许范围，不代表边界附近就是高概率区域。

比如：

- 真正拟合好的区域在中间
- 边界附近拟合都很差

那链自然大部分时间都待在中间，几乎不去边界。
这是正常的，而且恰恰说明后验分布集中在中间。

换句话说：

- `ub/lb` 是“允许去哪里”
- 后验分布决定“实际常去哪里”

这两个不是一回事。

**4. 为什么拒绝是必要的**

如果每个新点都接受，会发生什么？

那链就变成纯随机游走，只是在参数空间里乱逛。
这样采出来的就不是你想要的“后验分布”，而只是“提议分布”。

拒绝机制的作用是让链更愿意待在高后验区域。
所以接受/拒绝不是 bug，而是 MCMC 的核心。

你可以这样理解：

- 接受率高的区域 = 模型认为更像真的参数
- 接受率低的区域 = 模型认为解释数据的能力差

**5. 为什么原地不动的样本不完全没意义**

原地不动的样本不是“完全没意义”，但信息量比一个新位置小。

它仍然有意义，因为它表达了这件事：

- 在当前位置附近提了一个新点
- 但新点不如当前位置合理
- 所以模型更支持当前位置

也就是说，“重复出现”本身就在告诉你：这个区域概率比较高，链愿意停在这里。

但是为什么又说“有效样本数”会变少？

因为统计上我们更希望拿到很多“彼此不同、相对独立”的样本。
如果你 1000 个样本里有很多行都一样，比如：

```text
A
A
A
A
B
B
C
C
C
```

那虽然表面上有 9 个样本，但真正提供的新信息没有 9 份那么多。
这就是“有效样本数”比“总样本数”小的意思。

所以原地不动的样本：

- 不是没意义
- 但会降低采样效率

**6. 为什么要多条链**

多条链不是为了增加参数个数，而是为了检查：

- 不同起点出发
- 最后是不是都走到了同一个后验区域

如果 3 条链最后都在差不多的分布里活动，说明结果更可信。
如果 3 条链各走各的，说明可能还没收敛，或者后验结构很复杂。

**7. 一个非常直观的类比**

把参数空间想成一片山地：

- 高处 = 后验概率高
- 低处 = 后验概率低

MCMC 链像一个人在山地里乱走：

- 每次试着走一步
- 如果走到更高处，通常接受
- 如果走到更低处，有时也允许走，这样不容易卡死在局部小山头
- 走很久以后，这个人在哪些地方待得久，说明那些地方更“高”

你最后看的参数直方图，就是这个人长期活动轨迹的统计结果。

**8. 回到你的几个具体疑问**

你说“我想得到一个范围内所有参数的分布”：

- 对，但不是靠把整个范围均匀扫一遍
- 而是靠链在高概率区域停留更久，最终逼近这个分布

你说“往前移动是不是有时不成功”：

- 对
- 新参数总会生成
- 但不一定会被接受

你说“如果一步的 stepsize 没加上，我不就调用不到 ub 边界值了”：

- MCMC 不要求一定碰到边界
- 边界只是合法区间，不是必须访问的位置

你说“原地不动的样本没有意义吗”：

- 有意义，它说明当前位置比新提议点更可信
- 但重复太多会让样本相关性变强，降低效率

如果你愿意，我下一步可以直接给你画一个“2 个参数的简化链移动示意图”，用 5 到 8 步的小例子把“接受、拒绝、重复样本、最后形成分布”一次讲清楚。

> 那metropolis_hastings输出的是所有的attempts，但起始只用到了最后一个sample？因为按照你说的算法，最后一个sample才会出现在高概率区

不是。`metropolis_hastings` 输出的是整条链上的所有已记录样本，不是“只为了最后一个 sample”。

你这份代码后面真正用到的也不是最后一个样本，而是 `burn-in` 之后的所有样本。

在主程序 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:95>) 之后这段就是关键：

```matlab
burn_in = round(0.3 * n_samples);
samples_after_burnin = reshape(all_samples(:, burn_in+1:end, :), ...);
estimated_params = mean(samples_after_burnin, 1);
param_std = std(samples_after_burnin, 0, 1);
```

意思是：

- 前 30% 样本丢掉
- 后 70% 的样本全部拿来统计
- 用这些样本算后验均值、标准差、直方图、随机后验样本

所以不是“只用最后一个”。

你说“最后一个 sample 才会出现在高概率区”，这个说法也不准确。更准确的是：

- 链经过一段 burn-in 后，通常会进入高概率区
- 之后很多样本都会在高概率区附近波动
- 我们要的正是这一整段波动轨迹的统计分布

因为你要的是“分布”，不是“终点”。

如果只拿最后一个 sample，会有几个问题：

- 它只是一个点，不能代表整个后验分布
- 不能算标准差
- 不能画后验直方图
- 不能反映参数之间的不确定性
- 它还会有随机性，换一次链最后一个点可能不同

所以 MCMC 的思想不是：

- 一路走到最后，最后那个点就是答案

而是：

- 前面先让链走到稳态区域
- 后面把稳态区域里的大量样本当作后验分布的近似抽样

你可以把它理解成：

- `attempts` 是每次提议的新点
- `samples` 是每一步最终链停留的位置
- 真正统计用的是后半段很多个 `samples`
- 不是只看最后一个

再精确一点区分：

- `proposed_params`
  这是“尝试点”，有些会被拒绝，不一定进最终轨迹
- `samples(i,:)`
  这是第 `i` 步链真正所在的位置，才会进入输出矩阵
- `samples_after_burnin`
  这是后面真正拿来做参数分布统计的数据

所以 `metropolis_hastings` 输出的不是“所有 attempt”，而是“每一步链最终所在的位置”。
attempt 只是中间过程，只有被接受后才会成为新的位置；被拒绝的话，输出里记录的是旧位置再重复一次。

如果你愿意，我可以接着把 `proposed_params`、`current_params`、`samples(i,:)` 这三者在 [metropolis_hastings.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/metropolis_hastings.m:1>) 里再补一条特别直白的中文注释。

> 拿到了burn in 后的样本，模型是怎么继续算后验均值、标准差、直方图、随机后验样本的？

拿到 `burn-in` 之后的样本，后面的计算本质上都很直接：把这些样本当成“从后验分布里抽出来的一批近似样本”，然后做普通统计。

在你的主程序里，先做的是这一步，在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:95>) 附近：

```matlab
burn_in = round(0.3 * n_samples);
samples_per_chain = n_samples - burn_in;
samples_after_burnin = reshape(all_samples(:, burn_in+1:end, :), n_chains * samples_per_chain, []);
```

这段的意思是：

- 每条链前 30% 丢掉
- 保留每条链后 70%
- 再把 3 条链拼成一个二维矩阵 `samples_after_burnin`

这个矩阵的形状是：

- 行：后验样本编号
- 列：参数编号

也就是每一行都是一组完整参数：

```text
[sigma, alpha_infin, lambda, lambda_prime, k0_prime, sigma_noise]
```

然后下面几件事都是基于这个矩阵做的。

**1. 后验均值怎么来的**

在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:100>)：

```matlab
estimated_params = mean(samples_after_burnin, 1);
```

这里的 `mean(...,1)` 表示“按列求平均”。

意思就是：

- 第 1 列所有样本取平均，得到参数 1 的后验均值
- 第 2 列所有样本取平均，得到参数 2 的后验均值
- …
- 第 6 列所有样本取平均，得到参数 6 的后验均值

所以 `estimated_params` 是一个 `1 x 6` 向量。

你可以把它理解成：

“如果把 burn-in 后所有参数样本都看作这个参数可能取值的经验分布，那么均值就是这个经验分布的中心位置。”

**2. 后验标准差怎么来的**

紧接着在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:101>)：

```matlab
param_std = std(samples_after_burnin, 0, 1);
```

这里也是按列算标准差。

意思是：

- 第 1 列样本的标准差 = 参数 1 的后验不确定性
- 第 2 列样本的标准差 = 参数 2 的后验不确定性
- …
- 第 6 列样本的标准差 = 参数 6 的后验不确定性

标准差越大，表示这个参数在后验里分布越分散，不确定性越高。
标准差越小，表示样本更集中，参数估计更稳定。

**3. 直方图怎么来的**

在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:167>) 附近：

```matlab
histogram(samples_after_burnin(:, i), 50);
```

这里取的是第 `i` 个参数那一列的所有样本，然后分成 50 个 bin 画图。

意思是：

- 横轴：这个参数可能的取值范围
- 纵轴：样本落在每个区间里的次数

因为这些样本被当作来自后验分布，所以这个直方图就是该参数后验分布的经验近似。

也就是说：

- 某个区间柱子高，说明这个区间出现得多，后验概率更高
- 某个区间柱子低，说明这个区间出现得少，后验概率更低

**4. 随机后验样本怎么来的**

在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:126>) 附近：

```matlab
num_posterior_samples = 1000;
rand_idx = randperm(size(samples_after_burnin, 1), num_posterior_samples);
sampled_params = samples_after_burnin(rand_idx, :);
```

这段的意思是：

- 从 `samples_after_burnin` 的所有行里
- 随机抽 1000 行
- 每行还是一整组 6 个参数

所以 `sampled_params` 也是一个矩阵，只不过是从完整后验样本里随机抽出来的一部分。

它的作用不是再估计均值，而是为了后面的“不确定性传播”：

- 把每一组随机参数再送进 `jcal_reflection`
- 得到一条预测曲线
- 1000 组参数就得到 1000 条预测曲线
- 最后形成一团“后验预测云图”

也就是代码后面 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:215>) 开始的部分。

**5. 为什么这些操作都成立**

核心前提只有一句话：

`samples_after_burnin` 被看作是“来自后验分布的近似抽样”。

一旦接受这个前提，那么：

- 样本均值近似后验均值
- 样本标准差近似后验标准差
- 样本直方图近似后验概率密度形状
- 随机抽行就是从后验里再抽参数组合

所以后处理本身不神秘，难点其实在前面：

- 链有没有收敛
- burn-in 去得是否合理
- 样本是否足够多
- 自相关是否太强

如果这些前提还行，那么后面的均值、标准差、直方图就都只是普通统计。

你可以把整个流程压缩成一句话：

- `MCMC` 负责“造出后验样本”
- `mean/std/histogram/random draw` 负责“拿这些样本做统计总结和预测传播”

如果你要，我下一步可以直接把这一段整理成几行，补进 `working_history.md`，放在 `burn-in` 那段后面。

> 有几个问题，三条chain的结果应该都不一样，最终的结果是不是要判断一下每一条链是否收敛？如果是的话，收敛条件是什么？是每条链的最后几次结果与实验值做对比吗？还有，既然每条链的结果都不一样，他怎么把他们都合成一个samples_after_burnin一个二维矩阵，这个矩阵的行数跟每一条链的行数相等吗？

是，要判断每一条链是否已经“收敛到同一个后验分布附近”，这也是你代码里 `R-hat` 的作用。

不是看“每条链最后几次结果和实验值是否接近”来判断收敛。那样判断不对，因为 MCMC 的目标不是让最后一个点尽量贴实验值，而是让整条链在后验分布里稳定采样。

更准确地说，收敛看的是：

- 同一条链内部，后期波动是否稳定
- 不同链之间，是否都混到了相近的区域
- 各链对同一个参数给出的分布是否相似

你这份代码在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:84>) 到 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:93>) 用的是 `R-hat`。它比较的是：

- `W`：链内方差
- `B`：链间方差

如果不同链已经混合到同一个分布里，那么链间差异不该比链内波动大很多，这时 `R-hat` 会接近 `1`。

常见判断标准是：

- `R-hat` 非常接近 `1` 最好
- 通常 `< 1.1` 可勉强接受
- 更严格时希望 `< 1.05`
- 如果明显大于 `1.1`，一般说明还没收敛好

所以收敛条件不是“最后几次和实验值对比”，而是“不同链是否给出一致的后验采样结果”。

你问“3 条链结果都不一样，为什么还能合并”，关键就在这里：

- 前期不一样是正常的，因为初值不同
- 如果后期已经收敛，它们虽然不会一模一样，但会像从同一个分布里抽样出来的不同样本
- 这时就可以把它们合并

就像你从同一个正态分布里抽 3 批样本，这 3 批具体数字当然不同，但仍然可以合成一批更大的样本做统计。

至于 `samples_after_burnin` 怎么变成二维矩阵，看这段，在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:95>) 到 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:97>)：

```matlab
burn_in = round(0.3 * n_samples);
samples_per_chain = n_samples - burn_in;
samples_after_burnin = reshape(all_samples(:, burn_in+1:end, :), n_chains * samples_per_chain, []);
```

先看 `all_samples` 的维度：

- 第 1 维：`n_chains = 3`
- 第 2 维：每条链的样本数 `n_samples = 1000`
- 第 3 维：参数个数 `6`

所以 `all_samples` 是一个 `3 x 1000 x 6` 的三维数组。

去掉 30% burn-in 后：

- 每条链剩下 `700` 行
- 3 条链一共就是 `3 * 700 = 2100` 组参数样本

于是 `reshape(...)` 后的 `samples_after_burnin` 是：

- `2100 x 6`

也就是：

- 行数不等于“每条链的行数”
- 行数等于“3 条链剩余样本总数”
- 列数仍然是参数个数 `6`

所以如果单条链是 `700 x 6`，合并后三条链就是 `2100 x 6`。

可以直观写成：

```text
chain 1 after burn-in: 700 x 6
chain 2 after burn-in: 700 x 6
chain 3 after burn-in: 700 x 6
合并后:              2100 x 6
```

最后再强调一下：

- 链与链“不一样”不等于不能合并
- 只要它们已经收敛到同一个后验分布，就应该合并
- 合并是为了让后验统计更稳，样本更多

如果你愿意，我可以下一步把 `R-hat` 那几行公式在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:84>) 附近直接补上中文解释。

> 这个reshape公式的合并方法，排列方法是：chain1的第一行，chain2的第一行，chain3的第一行，然后chain1 的第二行,chain 2的第二行....这样子吗

不是按你写的那种“第一行交错穿插”的顺序去理解更稳妥。这里 `reshape` 是按 MATLAB 的列优先存储规则重排内存里的元素，不建议把它直观理解成手工一行一行交错拼接。

你这段代码里：

```matlab
samples_after_burnin = reshape(all_samples(:, burn_in+1:end, :), n_chains * samples_per_chain, []);
```

`all_samples(:, burn_in+1:end, :)` 的尺寸是：

```matlab
3 x 700 x 6
```

`reshape(..., 2100, 6)` 之后变成：

```matlab
2100 x 6
```

更关键的是：MATLAB 按列优先展开，所以前两维 `chain` 和 `sample index` 会先被揉在一起。
对于某一个固定参数列，顺序更接近于：

- chain 1 的第 1 个后 burn-in 样本
- chain 2 的第 1 个后 burn-in 样本
- chain 3 的第 1 个后 burn-in 样本
- chain 1 的第 2 个后 burn-in 样本
- chain 2 的第 2 个后 burn-in 样本
- chain 3 的第 2 个后 burn-in 样本
- …

所以如果你只看“某一列参数”，你的理解基本是对的：它确实是按链交错排列的。

但要小心一点，整行是 6 个参数一起走的，因此每一行仍然是一组完整参数：

```text
[sigma, alpha_infin, lambda, lambda_prime, k0_prime, sigma_noise]
```

也就是说第 1 行对应“某条链某一步”的整组参数，第 2 行对应下一条链同一步的整组参数，依此类推。

如果你想更直观、也更不容易误解，我其实建议把这句改成更明确的写法，比如先 `permute` 再 `reshape`，或者直接用循环拼接。因为现在这句虽然能用，但可读性一般。

最简单的直观理解可以记成：

- 合并后不是每条链各自一大段接在后面
- 而是同一采样步的不同链样本会交错排进去

如果你愿意，我可以下一步直接帮你在 `Uncertainty_quantification.m` 旁边补一条注释，把这个 `reshape` 的排列顺序写清楚。

> 我现在的输出是40张图，应该是按照8个porosity循环的，每个porosity对应五张图，你能帮我解释一下每一张图的含义吗，解释一组就行

可以。一组 `porosity` 会对应 5 张图，它们在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:152>) 之后依次画出来。你可以把它们理解成两类：

- 前 3 张：看 `MCMC` 和参数分布本身是否靠谱
- 后 2 张：看“这些参数”能不能解释实验数据，以及不确定性会带来多大预测散布

**1. Trace Plots**

这是第 1 张图，6 个子图，对应 6 个参数。
每个子图里有 3 条线，对应 3 条 chain，横轴是迭代次数，纵轴是该参数取值。

它主要看 3 件事：

- 链有没有在前期明显漂移，然后后期稳定下来
- 3 条链后期有没有混到差不多的区域
- 参数有没有老是卡在边界附近

图上还有：

- 一条竖虚线：`burn-in` 分界线
- 两条横虚线：该参数的 `lb` 和 `ub`

怎么看：

- 如果 3 条链后半段都在同一范围内上下波动，通常是好现象
- 如果某条链一直跟另外两条差很远，可能没收敛
- 如果曲线贴着上下界跑，说明边界可能太紧，或者参数识别有问题
- 如果几乎是一条很平的线，可能接受率太低，链没怎么动

**2. Histograms**

这是第 2 张图，也是 6 个子图，每个参数一个直方图。
它画的是 `burn-in` 之后该参数样本的分布形状。

图上还有：

- 黑色竖线：后验均值 `estimated_params(i)`
- 红色竖虚线：参数上下界 `lb(i)`、`ub(i)`

它回答的问题是：

- 这个参数最可能落在哪个范围
- 分布是窄还是宽
- 分布是单峰还是多峰
- 有没有撞到边界

怎么看：

- 分布很窄：说明参数比较确定
- 分布很宽：说明不确定性大
- 如果分布偏斜，说明均值未必能代表全部信息
- 如果有多峰，可能存在多个不同参数区间都能解释数据
- 如果峰贴近边界，说明边界在影响结果

**3. Std/Mean Ratio**

这是第 3 张图，是一个柱状图。
横轴是 6 个参数，纵轴是 `std / mean`。

它是一个简化版“相对不确定性”指标：

- 值小：说明标准差相对均值较小，参数更稳定
- 值大：说明这个参数相对不确定性更强

怎么看：

- 这是用于参数之间横向比较的，不是绝对标准
- 哪根柱子高，哪个参数就更难被精确识别
- 如果某个参数均值本身很接近 0，这个比值会显得特别敏感，解释时要小心

**4. Measured vs Predicted R and Alpha**

这是第 4 张图，最直接看“拟合效果”。
它有 3 个主要部分：

- 左上：实验 `R_exp` 实部 vs 模型预测 `R_predicted` 实部
- 右上：实验 `R_exp` 虚部 vs 模型预测 `R_predicted` 虚部
- 下方：实验吸声系数 `alpha_exp` vs 预测吸声系数 `alpha_predicted`，外加一条 `JCAL_O` 对照曲线

颜色含义通常是：

- 红点：实验数据
- 蓝线：用后验均值参数做出的预测
- 黑色虚线：`JCAL_O` 的对照结果

它回答的问题是：

- 这组反演参数能不能把反射系数的实部拟合好
- 能不能把虚部拟合好
- 最终换算出来的吸声系数是否也合理

怎么看：

- 蓝线越贴近红点，说明拟合越好
- 不能只看吸声系数，也要看实部和虚部，因为反射系数是复数
- 如果吸声拟合不错，但实部/虚部偏差大，说明可能只是模长凑巧接近
- 黑虚线和蓝线的差异反映“是否使用实际空气参数”对结果的影响

**5. Random Posterior Predictive Samples**

这是第 5 张图，也有 3 个主要部分。
它不是只画一条预测曲线，而是从后验里随机抽 1000 组参数，每组都做一次 `jcal_reflection`，于是形成很多条灰色曲线。

内容是：

- 左上：实验实部 + 一簇后验预测实部曲线
- 右上：实验虚部 + 一簇后验预测虚部曲线
- 下方：实验吸声系数 + 一簇后验预测吸声曲线

颜色含义通常是：

- 红点：实验值
- 灰线：不同后验样本对应的预测结果

它回答的问题是：

- 参数不确定性会导致多大的预测不确定性
- 实验点是不是大致落在这团后验预测云里面
- 哪些频段预测稳定，哪些频段预测分散

怎么看：

- 灰线很窄：说明这个频段预测稳定，不确定性小
- 灰线很散：说明这个频段对参数敏感，预测不确定性大
- 如果红点大部分落在灰色云团范围内，说明后验预测整体合理
- 如果红点长期跑在灰色云外面，说明模型或参数设定可能有系统性问题

你可以把这 5 张图记成一句话：

1. `Trace plot` 看链有没有收敛
2. `Histogram` 看参数后验分布长什么样
3. `Std/Mean` 看哪个参数最不确定
4. `Measured vs Predicted` 看后验均值拟合得好不好
5. `Posterior predictive` 看参数不确定性传到预测后会变成多大的曲线散布

如果你要，我下一步可以直接按你现在的某一个具体 `porosity` 图，结合图上的实线虚线红点灰线，再给你做一版“看图指南”。

> 在第五张图中，做了多少次后验随机样本的抽取？为什么我得到的左上和右上图是三条宽度差不多的灰色带状线？随机抽1000个参数的会抽成这样子吗？这跟三条链有关？

第 5 张图里，后验随机样本一共抽了 `1000` 组。

在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:126>) 这里先做了：

```matlab
num_posterior_samples = 1000;
rand_idx = randperm(size(samples_after_burnin, 1), num_posterior_samples);
sampled_params = samples_after_burnin(rand_idx, :);
```

然后在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:216>) 之后，把这 `1000` 组参数逐组送进 `jcal_reflection`，所以理论上是画了：

- `1000` 条实部预测曲线
- `1000` 条虚部预测曲线
- `1000` 条吸声预测曲线

你看到左上和右上像“三条宽度差不多的灰色带状线”，通常不是因为真的只抽到了 3 条，而是显示效果叠加后的视觉结果。

主要原因一般有这几个。

1. `plot(freq, R_real_samples, ...)` 是一次把整个矩阵都画出来。
`R_real_samples` 的尺寸是 `length(freq) x 1000`，MATLAB 会把每一列当一条线。1000 条半透明灰线叠在一起，很容易看起来像几条连续灰带，而不是 1000 条可分辨曲线。

2. 如果后验分布比较集中，1000 组参数给出的预测会很接近。
这样很多曲线几乎重合，就会自然形成一条“厚线”或“带状区域”。

3. 反射系数实部和虚部随频率通常有比较平滑的变化。
所以大量相近曲线叠起来，更容易形成几块宽度差不多的灰带，而不是杂乱无章的网。

这和“三条链”有关系，但不是你看到“三条灰带”的直接原因。

更准确地说：

- 3 条链共同提供了 `samples_after_burnin`
- 从这个合并后的后验样本池里再随机抽 1000 组参数
- 所以第 5 张图的灰线来源是“1000 个后验样本”
- 不是“每条链各画一条”

如果三条链确实都收敛到了相似分布，那么抽样后根本看不出哪条曲线来自哪条链。
只有在三条链没混合好、各自停在不同区域时，才可能在后验预测图里出现明显分成几簇的情况。

所以你现在看到“像三条灰带”，可能有两种解释：

- 更常见的情况：只是 1000 条相近曲线叠加后的视觉效果
- 另一种可能：后验样本确实有 2 到 3 个明显子群，导致预测也分成几簇

怎么区分这两种情况：

- 先看第 1 张 `trace plot`，3 条链后期是不是混在一起
- 再看第 2 张 `histogram`，参数分布是不是单峰
- 如果 `trace plot` 混得不错、`histogram` 也基本单峰，那第 5 张里的“几条灰带”大概率只是绘图叠加效果
- 如果 `trace plot` 里链分开、`histogram` 有多峰，那灰带分簇就可能是真的

还有一个很实际的点：你代码里的颜色写法

```matlab
plot(freq, R_real_samples, 'Color', [0.6 0.6 0.6 0.3]);
```

在不同 MATLAB 版本里，对 `plot` 的 RGBA 透明度支持并不稳定。
有些版本会忽略 alpha，只按灰色实线画，这样 1000 条线更容易糊成几条厚带。

所以结论是：

- 第 5 张图确实抽了 `1000` 组后验参数
- 看到灰色带状线是正常现象
- 这通常不是因为只剩 3 条链
- 而是很多预测曲线彼此接近、叠加后形成带状视觉效果
- 是否与 3 条链未收敛有关，需要结合 `trace plot` 和 `histogram` 一起判断

如果你愿意，我可以下一步直接帮你把第 5 张图改成更容易读的版本，比如：
- 只画 100 条随机曲线
- 再加一条后验均值曲线
- 或者改成画 5%/50%/95% 分位带，而不是 1000 条灰线全叠上去。

> 我还是没太懂sigma noise表达什么意思，你能给我讲一下在哪一步和哪一步之间用到了这个参数，这个参数是哪里来的？也是像其他参数一样randomly才出来然后往前走吗？

是的，`sigma_noise` 在这套代码里和前 5 个材料参数一样，也是 MCMC 里一起采样出来的参数；但它的含义不同。前 5 个参数是“材料本身的物理参数”，`sigma_noise` 是“统计误差参数”。

你可以把它理解成一句话：

“模型允许实验反射系数 `R_exp` 和预测反射系数 `R_predicted` 之间有多大的典型误差。”

它不是从实验文件里直接读出来的，也不是事先固定给定的，而是和别的参数一起放进

```matlab
params = [sigma, alpha_infin, lambda, lambda_prime, k0_prime, sigma_noise]
```

然后由 MCMC 去反演。

**它是从哪里来的**

在主程序 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:8>) 里，它就是第 6 个参数名 `Sigma Noise`。
在 [getFiberConstraints.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/getFiberConstraints.m:22>) 和 [getFiberConstraints.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/getFiberConstraints.m:29>) 里，它也有边界：

- 下界：`1e-6`
- 上界：`1`

所以它和其他参数一样，先有合法范围，再在这个范围里随机初始化。

在 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:74>) 生成初值时：

```matlab
init_params_all(chain, :) = lb + 0.2*(ub - lb) + 0.6*(ub - lb).*rand(...)
```

这里 6 个参数都一起随机初始化了，所以 `sigma_noise` 也是随机初值。

然后在 [metropolis_hastings.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/metropolis_hastings.m:24>) 提议新参数时：

```matlab
proposed_params = current_params + step_size .* randn(size(current_params));
```

这一步也是 6 个参数一起动，所以 `sigma_noise` 也是像其他参数一样“往前走”的。

**它在哪一步和哪一步之间被用到**

最关键的是在 [log_likelihood.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/log_likelihood.m:7>) 开始：

```matlab
sigma_noise = params(6);
```

然后流程是：

1. 当前或新提议的一组 `params` 被送进 `log_likelihood`
2. 取出前 5 个材料参数，送进 `jcal_reflection`
3. 得到预测反射系数 `R_predicted`
4. 把 `R_predicted` 和实验 `R_exp` 做差，得到残差
5. 用 `sigma_noise` 评价“这些残差算大还是算小”
6. 得到一个 `log_likelihood_value`
7. 再回到 `metropolis_hastings`，用这个值和旧参数的 `log_likelihood` 比较，决定接受还是拒绝

所以它被用在：

- `jcal_reflection` 算完预测结果之后
- `metropolis_hastings` 决定接受率之前

也就是说它不参与 JCAL 物理公式本身，它参与的是“预测和实验差多少时算合理”。

**为什么需要它**

因为实验数据不可能和模型完全一模一样。偏差可能来自：

- 仪器测量噪声
- `R.txt` 数据本身的小波动
- 模型简化误差
- 样品不均匀性
- 没建模进去的物理效应

`sigma_noise` 就是在说：

“我允许实验值和模型值之间有这么大的典型偏差。”

**它怎么影响接受/拒绝**

它不是直接的“接受概率”，但它会影响 `log_likelihood`，从而影响接受率。

如果 `sigma_noise` 小：

- 模型对残差很敏感
- 一点点误差都会被罚得很重
- 所以只有非常贴合实验的数据才会有高似然

如果 `sigma_noise` 大：

- 模型对残差更宽容
- 同样的误差看起来没那么严重
- 更多参数组合都可能被认为“还行”

所以它相当于一个“误差容忍尺度”。

**和其他参数最大的区别**

- 前 5 个参数改变的是 `R_predicted` 本身，因为它们进了 `jcal_reflection`
- `sigma_noise` 不改变 `R_predicted`
- 它只改变“如何评价 `R_predicted` 和 `R_exp` 的差异”

所以它不是材料物理属性，而是统计模型属性。

**一个很直观的类比**

把实验点和预测曲线的偏差想成“靶心周围的散布”：

- `sigma_noise` 小：你要求每一枪都非常接近靶心，容忍度低
- `sigma_noise` 大：你允许子弹打得更散一点，容忍度高

但如果你把 `sigma_noise` 设得太大，虽然什么都能解释，模型就失去区分力了。
这就是为什么对数似然里不仅有残差项，还有归一化项，会抑制它无限变大。

如果你愿意，我可以直接把这段解释整理成几条中文注释，补进 [log_likelihood.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/log_likelihood.m:1>) 和 [Uncertainty_quantification.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/Uncertainty_quantification.m:5>)。

> 这种把noise也搞成一个随机变动的值的原因是为了更高效找出最优的noise范围吗？比如说另一个思路，我完全可以把noise设为一个定值，然后看是不是太严格或者太宽松，然后再加大或者收紧我的noise容忍范围，这样不是更能满足我的要求吗，我想严格就可以严格想宽松就宽松

可以这么做，但那样你做的是“手动设定噪声水平的反演”，不是“把噪声不确定性也一起纳入后验推断”。

这两种思路都成立，只是目标不一样。

把 `sigma_noise` 也当成待估参数的原因，不主要是“更高效找最优 noise”，而是为了让模型自己从数据里判断：

- 实验和模型之间的典型误差量级大概多大
- 这个误差量级会怎样影响前 5 个材料参数的后验分布

也就是说，它是在回答：

“这些残差到底应该归因于参数没选对，还是归因于观测/模型噪声本来就存在？”

如果你把 `sigma_noise` 固定死：

- 你等于提前告诉模型“我认为误差就该是这么大”
- 后验分布会条件于这个假设
- 前 5 个参数的不确定性会直接受这个固定值影响

具体会这样：

- `sigma_noise` 设得很小：模型会被迫非常贴数据，参数后验通常更尖、更窄，但也更容易过拟合，或者链更难走
- `sigma_noise` 设得很大：模型会变宽松，参数后验通常更宽，很多参数组合都会显得“还行”

所以你说“我想严格就严格，想宽松就宽松”，这在工程上是能做的，但它更像是你在主观设定误差容忍规则，而不是让数据自己估这个规则。

两种方法的区别可以简单写成：

1. 固定 `sigma_noise`
- 适合你已经非常清楚测量误差水平
- 或者你就是想做一个带固定噪声假设的灵敏度分析
- 结果更依赖你的先验选择

2. 反演 `sigma_noise`
- 适合你并不确定实验误差到底多大
- 让数据自己推一个合理噪声尺度
- 参数不确定性会更完整，因为把噪声这部分也传播进去了

所以这不是“哪个更满足要求”的问题，而是“你的要求到底是什么”。

如果你的目标是：

- “我要一个我自己定义的严格标准”
那固定 `sigma_noise` 很合理。

如果你的目标是：

- “我要尽量少主观指定，让模型从数据自己判断误差尺度”
那把 `sigma_noise` 也当参数更合理。

实际上常见做法是两步都做：

- 先把 `sigma_noise` 当参数反演一次，看看数据支持的噪声量级大概在哪
- 再固定几个代表性的 `sigma_noise` 值做对比，检查结果对噪声假设有多敏感

这样你既有“数据驱动的噪声估计”，又有“人为严格/宽松标准下的敏感性分析”。

还有一点很关键：
如果你手动调 `sigma_noise`，本质上是在改模型的权重函数。它不只是“容忍度”这么简单，它会直接改变后验形状，所以不能把它看成一个纯显示参数。

一句话总结：

- 把 `sigma_noise` 设为随机参数，是为了让噪声尺度也由数据来估计
- 把它设为定值，是把噪声尺度当作你的外部假设
- 两者都可以，前者更偏统计推断，后者更偏人工控制

如果你要，我下一步可以直接帮你把当前代码改成“固定 `sigma_noise` 版本”，这样你就能和现在这版做一对一比较。

> 前三张图没有体现出在不同frequence下的表现吗？还是说MCMC采样的时候没有考虑frequence的差别？

前三张图没有把频率画在横轴上，但这不等于 MCMC 没考虑频率。

恰好相反，MCMC 在算 `log_likelihood` 的时候已经把整个频率范围内的数据都用进去了。只是前三张图展示的是“参数采样结果”，不是“按频率展开的拟合曲线”。

具体分开看：

1. 前三张图在看什么
- 第 1 张 `trace plot`：看每个参数在迭代过程中怎么变化
- 第 2 张 `histogram`：看每个参数的后验分布
- 第 3 张 `std/mean`：看每个参数的不确定性大小

所以这三张图的横轴分别是：
- 迭代步数
- 参数取值区间
- 参数编号

都不是频率。

2. MCMC 有没有考虑频率差别
有，而且是直接用整条频率曲线一起算的。

在 [log_likelihood.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/log_likelihood.m:24>) 之后，`jcal_reflection` 会用整组 `freq` 计算每个频率点上的预测反射系数 `R_predicted`。
然后在 [log_likelihood.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/log_likelihood.m:33>)：

```matlab
residuals = (R_exp_concat - R_predicted_concat).^2;
```

这里的残差不是一个频率点，而是所有频率点的实部和虚部一起比较。
最后在 [log_likelihood.m](</c:/Users/liuzi/OneDrive/Master Thesis/Fibers/log_likelihood.m:38>)：

```matlab
log_likelihood_value = -sum(residuals) / (2 * sigma_noise^2) ...
                       - N * log(sigma_noise * sqrt(2 * pi));
```

是把所有频率点的误差加总起来。

所以 MCMC 不是没看频率，而是：

- 每次试一组参数
- 用这组参数生成整条频率响应曲线
- 再把整条曲线和实验曲线一起比较
- 得到一个总分数 `log_likelihood`

3. 为什么前三张图看不出“哪个频率拟合得好”
因为前三张图是参数空间视角，不是频率响应视角。

真正体现不同频率表现的是后两张图：
- 第 4 张：实验和预测在各频率下的 `R`、`alpha` 对比
- 第 5 张：后验参数传播到各频率下后的预测散布

所以如果你想看“低频拟合好不好、高频拟合好不好”，应该重点看第 4、5 张，不是前 3 张。

4. 一个更准确的理解
前三张图回答的是：
- 参数有没有收敛
- 参数分布长什么样
- 参数不确定性大不大

后两张图回答的是：
- 这些参数在不同频率下能不能解释实验
- 参数不确定性会在不同频率下造成多大预测不确定性

如果你愿意，我可以继续帮你解释一下：现在这套 `log_likelihood` 其实默认把所有频率点“同等对待”，如果你想让某些频段更重要，代码该怎么改。
