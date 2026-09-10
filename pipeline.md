# Ordinary World Model 完整算法流程

## 1. 方法目标

本方法面向视觉 Sokoban 环境。它首先从真实图像中学习可用于预测和决策的 latent
状态，然后利用 World Model 在 latent 空间生成想象轨迹，最后采用 PPO 训练策略。

完整过程分为六个阶段：

1. 收集真实环境轨迹，并提取 Qwen 视觉特征；
2. 训练 observation-grounded posterior 和 action-conditioned prior；
3. 在独立验证集上检查 World Model 的空间语义和多步预测能力；
4. 固定 World Model，单独训练并校准 Reward Head；
5. 在正式 PPO 前完成 Critic warmup；
6. 使用 World Model 生成想象轨迹，并按照标准 PPO 更新 Actor 和 Critic。

普通 World Model 不预测下一帧 RGB，也不直接生成显式棋盘。环境状态由 36 个
保持空间顺序的 latent slots 表示，状态转移在 latent 空间完成。

---

## 2. 整体 Pipeline

| 阶段 | 输入 | 核心操作 | 输出 |
|---|---|---|---|
| 数据准备 | 真实 RGB episode | Qwen 视觉编码、时间对齐、数据划分 | Tokenized replay |
| World Model 训练 | 真实状态、动作和下一状态 | Posterior grounding、prior prediction、语义监督 | World Model checkpoint |
| World Model 审计 | Held-out episode | 空间解码、动作敏感性、多步预测检查 | 通过门控的 World Model |
| Reward Head 训练 | 多步 prior endpoint | 成功分类、阈值校准 | 校准后的 Reward Head |
| Critic warmup | 固定状态与回报目标 | 只训练价值函数 | 可用于 PPO 的 Critic |
| PPO | 当前策略产生的想象轨迹 | GAE、裁剪策略目标、价值回归 | 更新后的 Actor 和 Critic |
| World Model 刷新 | 新收集的真实 episode | 周期性监督学习和验证 | 更新后的想象环境 |
| 真实环境评估 | 固定真实关卡 | 执行当前策略 | Success Rate 等指标 |

数据在系统中沿三条路径流动：

1. **真实观测路径**：RGB 图像经过 Qwen 视觉编码器和 posterior，形成与真实局面
   对齐的 latent belief；
2. **模型想象路径**：从真实 belief 出发，prior 根据动作递归预测未来 belief；
3. **策略优化路径**：Actor 在想象轨迹中选择动作，Critic 估计价值，PPO 根据
   advantage 更新二者。

World Model 使用真实环境转移进行监督学习。PPO 使用 World Model 生成的数据，
但 PPO 损失不直接反向传播到 World Model。

---

## 3. 数据与时间轴

设一条 episode 包含 \(T\) 次真实环境转移：

\[
o_0 \xrightarrow{a_0,r_0} o_1
\xrightarrow{a_1,r_1}\cdots
\xrightarrow{a_{T-1},r_{T-1}} o_T.
\]

其中：

- \(o_t\) 是时刻 \(t\) 的真实图像观测；
- \(a_t\) 是在 \(o_t\) 下执行的动作；
- \(r_t\) 是执行 \(a_t\) 后得到的环境奖励；
- \(o_T\) 是最后一次动作后的终止观测。

包含 \(T\) 次转移的 episode 必须保存 \(T+1\) 个观测。动作和奖励在存储时也补齐
到相同长度，最后一项只是占位符，不参与训练。

主要张量及其含义如下：

| 符号 | 形状 | 含义 |
|---|---:|---|
| \(O\) | \(B\times(T+1)\times K\times D\) | Qwen 视觉 token 序列 |
| \(A\) | \(B\times(T+1)\) | 动作序列，最后一项为占位符 |
| \(R\) | \(B\times(T+1)\) | 奖励序列，最后一项为占位符 |
| \(Z_t\) | \(B\times K\times D\) | 时刻 \(t\) 的 belief state |
| \(\ell_t\) | \(B\) | Reward Head 输出的成功 logit |

其中 \(B\) 是 batch size，\(K=36\) 是 belief slot 数量，\(D\) 是 Qwen hidden
dimension。

时间对齐采用：

\[
Z_t^{post}=f_{post}(Z_{t-1}^{post},o_t),
\]

\[
Z_t^{pri}=f_{pri}(Z_{t-1}^{post},a_{t-1}),
\]

\[
y_t=\mathbb{1}[r_{t-1}>\tau_r].
\]

时刻 \(t\) 的 belief 表示执行 \(a_{t-1}\) 后到达的状态。初始状态 \(Z_0\)
没有对应的前一奖励，所以不参与 transition reward supervision，但仍可参与
posterior representation regularization。

---

## 4. Qwen 视觉状态表示

每张 RGB 图像经过 Qwen2.5-VL 的图像预处理器、视觉编码器和 visual merger，得到
与 Qwen language hidden space 对齐的图像特征。

由于不同输入图像可能产生不同数量的原生视觉 token，系统将二维视觉特征重采样
到固定的 \(6\times6\) 网格：

\[
o_t\in\mathbb{R}^{36\times D}.
\]

这种处理保留视觉 token 的二维空间顺序。第 \(k\) 个 token 始终对应相对固定的
图像区域，使玩家、箱子、目标和墙体等空间信息能够进入后续 belief slots。

当前 Qwen2.5-VL-3B 配置为：

| 项目 | 数值 |
|---|---:|
| 视觉 token 数 | 36 |
| Belief slot 数 | 36 |
| Qwen hidden dimension | 2048 |
| 环境动作数 | 4 |
| Null action 数 | 1 |

Qwen 视觉 token 是 posterior 的观测输入。V-JEPA 特征仅作为训练期间的独立语义
监督目标，不与 Qwen token 拼接，也不在策略推理时提供额外输入。

---

## 5. Belief State

World Model 使用有序 slot 集合表示状态：

\[
Z_t=[z_t^1,z_t^2,\ldots,z_t^K],
\qquad z_t^k\in\mathbb{R}^{D}.
\]

这些 slots 不被显式指定为玩家、箱子或墙体。它们通过视觉位置对齐、状态转移
监督和语义监督，自行学习保留棋盘结构。

每条 episode 从共享的可学习初始状态开始：

\[
Z_{-1}=Z_{init}.
\]

第一次 posterior inference 使用一个保留的 null action，表示初始观测之前没有
发生真实环境动作。

---

## 6. Posterior 与 Prior

### 6.1 Posterior

Posterior 根据当前真实观测建立 grounded belief：

\[
Z_t^{post}=f_{post}(Z_{t-1}^{post},o_t).
\]

它相当于状态估计器，将历史 belief 与当前真实图像结合，使 latent state 重新对齐
到真实环境。

当前 Qwen 配置让 posterior 主要由视觉观测决定，并用较小的 recurrent residual
保留历史信息：

\[
Z_t^{post}=\operatorname{LN}
\left(\operatorname{LN}(o_t)+s_r\widetilde Z_t\right),
\qquad s_r=0.25.
\]

当前 posterior 不使用真实动作作为条件。这可以减少模型根据动作猜测当前状态的
机会，使 posterior 更接近由当前图像决定的状态坐标。

### 6.2 Prior

Prior 不读取未来真实观测，只根据上一状态和动作预测下一状态：

\[
Z_t^{pri}=f_{pri}(Z_{t-1},a_{t-1}).
\]

它是想象轨迹中的实际状态转移模型。PPO rollout 开始后，后续 belief 均由 prior
递归产生，因此策略训练质量主要取决于 prior 的动作敏感性和多步稳定性。

### 6.3 共享转移网络

Posterior 和 prior 共享 Qwen transition backbone。该骨干支持视觉、动作和 belief
三类 token，并加入相应的类型表示及环境表示。当前 action-free posterior 接收
视觉观测和历史 belief，动作位置统一使用 null action；prior 接收真实动作和历史
belief。

Qwen 输出中与 belief slots 对齐的 hidden states 被解释为候选状态变化
Δ_t。最终 belief 采用门控残差更新：

\[
g_t=\sigma(W_g\Delta_t+b_g),
\]

\[
\widetilde Z_t=\operatorname{LayerNorm}
\left(Z_{t-1}+g_t\odot\Delta_t\right).
\]

门的初始值约为 0.12，使模型在训练初期只对历史 belief 做较小修改，降低递归状态
快速发散的风险。

---

## 7. World Model 监督训练

### 7.1 Truncated BPTT

完整 episode 被切分为长度 \(H\) 的不重叠窗口。第一个窗口从 learned initial
belief 开始；后续窗口接收上一窗口最后一个 posterior belief，但在窗口边界停止
梯度。

窗口内部依次计算 posterior 和 prior。每个 prior 与同一时刻、同一起点产生的真实
posterior 对齐。基础 dynamics supervision 因而是 teacher-forced 的单步预测。

### 7.2 Latent dynamics loss

Prior 预测停止梯度后的 posterior target：

\[
L_{dyn}=
\frac{1}{\sum_t m_tw_t}
\sum_t m_tw_t
\left\|Z_t^{pri}-\operatorname{sg}(Z_t^{post})\right\|_2^2.
\]

其中 \(m_t\) 排除初始状态和 padding，\(w_t\) 是窗口内的时间衰减权重。若启用
EMA teacher，target 可以由参数指数滑动平均后的 posterior 提供。

### 7.3 Recursive open-loop loss

当启用 open-loop supervision 时，模型从一个真实 grounded posterior 出发，然后
只使用动作递归预测未来 belief：

\[
\widehat Z_{t+k}^{pri}
=f_{pri}(\widehat Z_{t+k-1}^{pri},a_{t+k-1}).
\]

第 \(k\) 步预测与对应的真实 posterior 比较：

\[
L_{open}=\sum_{k=1}^{H_o}
\alpha_o^{k-1}
\left\|\widehat Z_{t+k}^{pri}
-\operatorname{sg}(Z_{t+k}^{post})\right\|_2^2.
\]

该目标直接训练误差累积后的状态恢复能力。需要特别区分：BPTT 窗口长度大于 1
并不自动等于多步 open-loop；只有前一步 prior 被继续用作下一步输入时，才形成
真正的递归预测。

### 7.4 Action-aware supervision

仅使用 absolute latent MSE 时，模型可能通过复制当前状态或预测平均变化获得较低
损失。为增强动作敏感性，可以比较真实变化和预测变化：

\[
\Delta_t^{post}=\operatorname{sg}(Z_t^{post}-Z_{t-1}),
\qquad
\Delta_t^{pri}=Z_t^{pri}-Z_{t-1}.
\]

变化方向损失为：

\[
L_{delta}=1-\cos(\Delta_t^{pri},\Delta_t^{post}).
\]

还可以从完整的有序 slot change 中预测真实动作，形成 inverse-action
classification loss：

\[
L_{inv}=\operatorname{CE}(q(\Delta_t),a_{t-1}).
\]

只有真实状态变化超过阈值的 transition 才进入上述目标，避免静止或无效动作提供
不可靠的方向监督。

### 7.5 V-JEPA semantic teacher

训练期间，每个真实画面额外具有一组冻结的 V-JEPA 语义特征：

\[
V_t\in\mathbb{R}^{36\times1408}.
\]

一个逐 slot 的线性映射将 belief 投影到 teacher 空间。主要包含三个目标。

Prior future-state loss：

\[
L_{vjepa}^{pri}=1-cos(P_v(Z_t^{pri}),V_t).
\]

Posterior grounding loss：

\[
L_{vjepa}^{post}=1-cos(P_v(Z_t^{post}),V_t).
\]

Action-sensitive delta loss：

\[
L_{vjepa}^{delta}=1-cos\left(
P_v(Z_t^{pri})-\operatorname{sg}(P_v(Z_{t-1}^{post})),
V_t-V_{t-1}
\right).
\]

三个目标都逐 slot 计算，不进行全局平均池化。Teacher 特征始终停止梯度，也不会
成为 World Model 或策略在推理时的输入。

### 7.6 Anti-collapse regularization

为避免所有 belief 收缩到少量相似表示，训练对有效 posterior beliefs 加入：

- 随机投影上的分布正则；
- 每个 latent dimension 的方差下限；
- 可选的维度间 covariance penalty。

这些正则在一个训练 batch 的全部 BPTT 窗口处理完成后统一计算一次，以避免窗口
数量改变正则项的相对权重。

### 7.7 总损失

完整的 World Model 目标为：

\[
\begin{aligned}
L_{WM}={}&L_{dyn}
+\lambda_oL_{open}
+\lambda_{delta}L_{delta}
+\lambda_{inv}L_{inv}\\
&+\lambda_{vp}L_{vjepa}^{pri}
+\lambda_{vpost}L_{vjepa}^{post}
+\lambda_{vd}L_{vjepa}^{delta}
+L_{regularization}.
\end{aligned}
\]

各项先按照自己的有效样本数归一化，再乘相应系数。一个 batch 中所有监督项和
正则项合并后只进行一次参数更新。

---

## 8. 当前 Stage-1 配置

当前 Qwen + V-JEPA Teacher 训练采用：

| 项目 | 设置 |
|---|---:|
| Backbone | Qwen2.5-VL-3B |
| Belief slots | 36 |
| Hidden dimension | 2048 |
| Attention | Bidirectional |
| Action condition | Learned embedding |
| Posterior grounding | Visual anchor |
| Posterior recurrent scale | 0.25 |
| BPTT horizon | 2 |
| Open-loop horizon | 0 |
| V-JEPA prior coefficient | 0.50 |
| V-JEPA posterior coefficient | 0.10 |
| V-JEPA delta coefficient | 0.25 |

该阶段的主要目标为：

\[
L_{stage1}=L_{dyn}
+0.50L_{vjepa}^{pri}
+0.10L_{vjepa}^{post}
+0.25L_{vjepa}^{delta}
+L_{regularization}.
\]

这里的 BPTT horizon 为 2，表示每个窗口含两个连续的 teacher-forced target。
由于 open-loop horizon 为 0，这个设置本身不构成递归两步训练。递归 H2 或 H4
能力需要由后续审计或专门的 prior refinement 阶段验证。

---

## 9. World Model Release Gate

较低的 latent loss 不能单独证明 World Model 学到了正确的 Sokoban 状态转移。
例如，复制当前状态在大量未发生明显变化的 transition 上也可能取得较低误差。

因此，训练完成后需要在固定 held-out episode 上执行独立审计。审计使用轻量空间
decoder，从有序 belief slots 中预测：

- 玩家位置；
- 箱子位置；
- 目标位置；
- 墙体布局。

随后分别评价：

1. Posterior 是否保留真实棋盘的空间结构；
2. Logged-action prior 是否预测正确的下一状态；
3. 同一起点下不同动作是否产生有差异的合理结果；
4. 状态真正发生变化的样本上，prior 是否优于复制状态 baseline；
5. 递归 H1、H2 和更长 horizon 的误差是否处于允许范围。

只有通过空间语义、动作敏感性和多步预测门控的 World Model，才进入后续 Reward
Head 和强化学习阶段。

---

## 10. Reward Head

### 10.1 目标定义

Reward Head 判断一个 predicted belief 是否对应任务成功：

\[
\ell_t=h_r(Z_t),
\qquad
p_t=\sigma(\ell_t).
\]

其中 \(p_t\) 是成功概率。Reward Head 是成功分类器，不是一般的连续奖励模型。

Reward Head 直接读取完整的有序 belief slots，不先将它们汇聚为单个特征向量。设
endpoint belief 为

\[
Z_t=[z_t^1,z_t^2,\ldots,z_t^K]\in\mathbb{R}^{K\times D}.
\]

Reward Head 先对每个 slot 独立使用共享的低维投影：

\[
u_t^k=\phi(z_t^k),
\qquad u_t^k\in\mathbb{R}^{d_r}.
\]

随后按照原始空间索引拼接全部低维 slot 表示，并由位置敏感的分类器输出 logit：

\[
x_t=[u_t^1;u_t^2;\ldots;u_t^K]\in\mathbb{R}^{K d_r},
\qquad
\ell_t=h_{cls}(x_t).
\]

逐 slot 投影用于控制参数量，有序拼接保证每个空间位置仍具有独立坐标，使分类器能够判断玩家、箱子和目标之间的组合
关系。训练标签只来自与 endpoint 精确对齐的真实 transition reward：

\[
y_t=\mathbb{1}[r_{t-1}>\tau_r].
\]

### 10.2 多 Horizon 特征

固定 World Model 后，从真实 grounded posterior 出发，沿真实动作序列执行多个
horizon 的 prior rollout。每个 rollout endpoint 的完整有序 belief tensor 与对应
真实 success label 组成 Reward Head 数据。训练缓存也保留全部 slots 及其顺序，
不提前压缩为单个向量。

这样可以同时训练 Reward Head 识别一步和多步 prior endpoint，减少它只适应某个
固定想象长度的风险。

### 10.3 数据划分与阈值校准

Reward Head 使用互不重叠的四组数据：

1. Training split 用于参数更新；
2. Validation split 用于选择模型；
3. Calibration split 用于选择成功概率阈值；
4. Official held-out split 只用于最终门控。

由于成功样本稀少，训练采用带正类权重的二分类交叉熵：

\[
L_{reward}=\operatorname{BCEWithLogits}(\ell_t,y_t;w_{pos}).
\]

阈值确定后，在 official split 上检查 ROC-AUC、average precision、precision、
recall、false-positive rate 和 Brier score，同时检查各个 rollout horizon 的表现。

通过门控后，Reward Head 参数和校准阈值与 World Model 一起用于 imagined rollout。

---

## 11. Critic Warmup

正式 PPO 开始前，Actor 暂不更新，先训练 Critic。这样可以避免随机初始化的价值
函数产生随机 advantage，进而在最初几轮 PPO 中把策略推向错误方向。

Critic warmup 的流程为：

1. 从真实 replay 中选择 grounded belief；
2. 构造对应的回报或动作价值监督；
3. 将样本划分为训练 replay 和固定 validation panel；
4. 只优化 Critic 的价值回归目标；
5. 在 validation panel 上计算 explained variance 和均方误差；
6. 当 Critic 持续达到预设标准后，结束 warmup 并释放 Actor。

Critic warmup 是 PPO 之前的初始化阶段。完成 warmup 后，Actor 和 Critic 进入标准
PPO 的同步数据采样与更新过程。

---

## 12. 标准 PPO 阶段

### 12.1 基本原则

完成 Critic warmup 后，每一轮 PPO 都遵循标准的 on-policy 更新逻辑：

1. 固定当前策略作为行为策略；
2. 使用该策略收集一批完整的想象轨迹；
3. 根据轨迹奖励和旧价值估计计算 advantage 与 return；
4. 在同一批数据上执行有限轮 minibatch PPO 更新；
5. 丢弃旧轨迹，再用更新后的策略收集下一批数据。

这里的 on-policy 是相对于当前 World Model 中的想象环境而言。每轮 PPO 数据都由
更新前的当前策略产生，不能在策略更新后继续无限复用。

### 12.2 想象轨迹的初始状态

每条想象轨迹首先从真实 replay 中选择一个时刻。系统从 episode 初始观测开始递推
posterior，直到该时刻，得到真实观测支持的 grounded belief：

\[
Z_0^{imag}=Z_t^{post}.
\]

这种初始化使想象轨迹从真实数据分布附近开始，降低从任意 latent 状态启动造成的
分布偏移。

### 12.3 使用当前策略采样轨迹

设当前行为策略为 \(\pi_{old}\)。在每个想象时刻 \(h\)，依次执行：

\[
a_h\sim\pi_{old}(\cdot\mid Z_h),
\]

\[
Z_{h+1}=f_{pri}(Z_h,a_h),
\]

\[
\ell_h=h_r(Z_{h+1}),
\]

\[
\widetilde r_h=g(\ell_h).
\]

其中 \(g\) 将校准后的成功 logit 转换为标量奖励。采样时保存：

- 当前 belief \(Z_h\)；
- 动作 \(a_h\)；
- 旧策略对该动作的对数概率；
- Critic 的旧价值估计；
- 想象奖励 \(\widetilde r_h\)；
- 终止标记和有效步 mask。

World Model 和 Reward Head 在轨迹生成及 PPO 更新期间保持参数不变，也不接收
PPO 梯度。

### 12.4 Reward 与终止

如果 Reward Head 只被校准为固定 horizon 的 endpoint success classifier，则想象
轨迹应采用固定长度：中间步骤不因成功 logit 而提前终止，最后一步根据成功概率
给出 endpoint reward。

如果 Reward Head 已被独立验证为逐 transition 的终止检测器，则可以在每一步使用
同一成功阈值。预测成功时设置终止标记，后续位置作为 padding，并从 PPO batch
中移除。

Reward 定义和终止规则必须一致，避免一个状态被判定为终止成功却得到非正奖励，
或者未达到成功阈值却得到正奖励。

### 12.5 Bootstrap

对于因固定 rollout horizon 截断、但没有终止的轨迹，可以使用 Critic 对末端状态
进行 bootstrap：

\[
V_{H}=V_\phi(Z_H).
\]

对于真正终止的轨迹，末端 continuation value 必须为零。若不启用 bootstrap，
所有想象片段末端统一使用零值。

### 12.6 Generalized Advantage Estimation

使用轨迹生成时保存的旧价值估计计算 temporal-difference residual：

\[
\delta_h=
\widetilde r_h+\gamma(1-d_h)V_{old}(Z_{h+1})-V_{old}(Z_h).
\]

随后从后向前计算 GAE：

\[
\widehat A_h=
\delta_h+\gamma\lambda(1-d_h)\widehat A_{h+1}.
\]

价值回归目标为：

\[
\widehat R_h=\widehat A_h+V_{old}(Z_h).
\]

若轨迹包含 padding，mask 在反向递推内部生效，使无效步骤的 advantage 为零，并
阻止其影响前面的有效步骤。进入 PPO 前，可以在整个有效 batch 上对 advantage
进行标准化。

### 12.7 Clipped policy objective

设更新后的策略为 \(\pi_\theta\)，概率比为：

\[
\rho_h(\theta)=
\frac{\pi_\theta(a_h\mid Z_h)}
{\pi_{old}(a_h\mid Z_h)}.
\]

标准 PPO clipped surrogate objective 为：

\[
L_{clip}(\theta)=
\mathbb{E}_h\left[
\min\left(
\rho_h(\theta)\widehat A_h,
\operatorname{clip}(\rho_h(\theta),1-\epsilon,1+\epsilon)
\widehat A_h
\right)
\right].
\]

优化时最小化其相反数。Clipping 限制单轮更新中策略概率发生过大的变化。

### 12.8 Value objective

Critic 使用同一批 on-policy 轨迹的 return target 做价值回归：

\[
L_V(\phi)=
\mathbb{E}_h\left[
\left(V_\phi(Z_h)-\widehat R_h\right)^2
\right].
\]

Warmup 结束后，Critic 与 Actor 在每轮 PPO 中共同更新。Critic 的作用是降低策略
梯度方差，并为非终止的 rollout endpoint 提供 bootstrap value。

### 12.9 Entropy bonus 与总目标

为了维持必要的探索，可以加入策略熵：

\[
\mathcal H(\pi_\theta)=
-\mathbb{E}_h\sum_a
\pi_\theta(a\mid Z_h)
\log\pi_\theta(a\mid Z_h).
\]

标准 PPO 总损失为：

\[
L_{PPO}=
-L_{clip}
+c_VL_V
-c_H\mathcal H(\pi_\theta).
\]

其中 \(c_V\) 控制价值损失权重，\(c_H\) 控制熵奖励权重。

### 12.10 Minibatch 更新

一批想象轨迹经过 GAE 后被展平成有效 transition 集合。随后：

1. 随机打乱全部有效 transition；
2. 划分为若干 minibatches；
3. 在同一批数据上训练有限个 epoch；
4. 每个 minibatch 重新计算新策略概率和当前价值；
5. 使用保存的旧策略概率计算 PPO ratio；
6. 对梯度进行裁剪并更新 Actor 和 Critic；
7. 完成本轮后丢弃该批数据。

可以监控近似 KL divergence、clip fraction、policy entropy、value loss 和 explained
variance。如果一次更新使新旧策略差异超过目标范围，可以提前结束本轮剩余
minibatch，以保持 PPO 的小步更新特征。

---

## 13. 周期性 World Model 刷新

World Model 刷新位于相邻 PPO 迭代之间，不改变单轮 PPO 的标准优化过程。

每到刷新周期，执行：

1. 从 offline replay 和当前策略新收集的 online replay 中采样真实 episode；
2. 暂停 Actor 和 Critic 更新；
3. 使用 World Model 的 Phase-1 监督目标执行若干梯度更新；
4. 在固定 held-out replay 上重新评价 latent dynamics、空间语义和多步预测；
5. 接受通过验证的 World Model，并用于下一轮想象轨迹；
6. 若验证明显退化，则恢复刷新前的模型和优化器状态。

刷新过程只使用真实 transition supervision，不使用 PPO loss 更新 World Model。
完成刷新后，新的 PPO 迭代重新采集轨迹，因此这些轨迹仍由当前策略和当前版本的
想象环境产生。

---

## 14. 真实环境评估

想象回报只反映策略在 learned model 中的表现，不能代替真实环境结果。因此需要
定期在固定且与训练数据隔离的真实 Sokoban 关卡上评价当前策略。

主要指标包括：

- 成功率；
- 平均 episode return；
- 平均解题步数；
- 动作分布和策略熵；
- 相对于训练前策略的提升；
- 不同评估轮次之间的稳定性。

评估时，真实图像经过相同的 Qwen tokenizer 和 posterior grounding。Actor 根据
真实 posterior belief 选择动作，环境返回下一张真实图像。Prior 只参与训练时的
想象，不替代真实评估中的环境转移。

---

## 15. 端到端算法流程

### 阶段一：准备数据

1. 在真实 Sokoban 环境中收集 RGB episode；
2. 保留每次动作前后的完整观测，使 \(T\) 次转移对应 \(T+1\) 张图像；
3. 使用 Qwen 视觉编码器提取 36 个有序视觉 token；
4. 为训练阶段保存逐帧 V-JEPA teacher target；
5. 按 episode 或关卡划分训练集与 held-out 集。

### 阶段二：训练 World Model

1. 从训练集采样真实 episode；
2. 按 BPTT horizon 切分序列；
3. 使用真实观测建立 posterior belief；
4. 使用上一 posterior 和真实动作预测 prior belief；
5. 计算 latent dynamics、V-JEPA teacher、动作变化和防坍缩目标；
6. 聚合全部窗口损失并更新 World Model；
7. 周期性在 held-out 数据上评价训练状态。

### 阶段三：审计 World Model

1. 从 posterior 和 prior slots 解码玩家、箱子、目标和墙体；
2. 评价真实动作的一步状态预测；
3. 比较同一起点下不同动作的预测结果；
4. 检查发生真实变化的样本是否优于复制状态 baseline；
5. 评价递归 H1、H2 和更长 horizon；
6. 只保留通过全部必要门控的 checkpoint。

### 阶段四：训练 Reward Head

1. 固定 World Model；
2. 从真实 posterior 起点生成多 horizon prior endpoints；
3. 保存每个 endpoint 的完整有序 belief slots；
4. 使用精确对齐的 transition success 作为标签；
5. 直接使用有序 slots 训练位置敏感的成功分类器；
6. 在独立 calibration split 上确定成功阈值；
7. 在 official held-out split 上完成最终门控。

### 阶段五：Critic warmup

1. 固定 Actor；
2. 为 grounded beliefs 构造回报监督；
3. 只训练 Critic；
4. 在固定 validation panel 上评价价值拟合；
5. Critic 稳定达到标准后释放 Actor。

### 阶段六：标准 PPO

1. 固定当前策略作为本轮行为策略；
2. 从真实 replay 中采样并 posterior-ground 想象起点；
3. 使用当前策略、prior 和 Reward Head 生成一批想象轨迹；
4. 保存旧动作概率、旧价值、奖励和终止标记；
5. 使用 GAE 计算 advantage 和 return；
6. 使用 clipped surrogate 更新 Actor；
7. 使用 return target 更新 Critic；
8. 加入 entropy bonus，并在有限轮 minibatch 上优化；
9. 丢弃本轮轨迹，进入下一轮 on-policy 数据采样；
10. 周期性执行真实环境固定关卡评估。

### 阶段七：刷新与继续训练

1. 在设定的 PPO 间隔收集新的真实策略数据；
2. 在相邻 PPO 迭代之间监督刷新 World Model；
3. 在固定 held-out replay 上验证刷新结果；
4. 接受通过验证的模型，或恢复刷新前状态；
5. 使用当前 World Model 和当前策略重新采集下一轮 PPO 轨迹。

---

## 16. 关键算法认识

1. **Posterior 和 prior 承担不同职责。** Posterior 建立真实状态坐标，prior 才是
   PPO 想象阶段实际使用的环境动力学。

2. **BPTT horizon 不等于 open-loop horizon。** 连续处理多个真实观测仍可能只是
   多个单步 teacher-forced prediction；递归 prior 才能检验真正的多步预测能力。

3. **低 latent MSE 不是充分条件。** 复制状态、动作无关预测和 representation
   collapse 都可能取得较低 MSE，因此需要空间语义和动作敏感性审计。

4. **Reward Head 是成功分类器。** 它的输出需要校准并转换成 PPO reward。分类
   准确不自动意味着对所有动作的长期价值排序正确。

5. **Critic warmup 只负责提供可靠初始化。** Warmup 完成后，PPO 阶段仍按照标准
   Actor-Critic PPO 同步更新，而不是继续采用独立的特殊 Critic 训练流程。

6. **每轮 PPO 必须重新采样。** PPO 数据由更新前策略产生，只能用于有限轮更新；
   策略或 World Model 改变后都应重新生成想象轨迹。

7. **真实环境评估不可替代。** 想象 return 的提升可能来自 World Model bias，最终
   策略效果必须由固定真实关卡上的成功率确认。
