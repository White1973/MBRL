# Qwen + V-JEPA Teacher MBRL 最新实验分析

> 记录日期：2026-09-03
> 实验目录：`MBRL0901`  
> 主要实验：`qwen_vjepa_teacher_seed1`  
> 本文是当前实验状态的快照。Stage 1–3 已完成验收；Stage 4 PPO 仍在运行，不能把其中间结果写成最终结论。

## 1. 当前结论

目前已经打通并通过质量 Gate 的部分是：

1. 基于原生 Qwen2.5-VL 空间 token 的 World Model 表征；
2. 使用冻结 V-JEPA 特征作为训练期空间语义教师；
3. 经过 position-aware prior repair 后的动作条件动力学；
4. Reward Head 的验证集、校准集和独立测试集验收；
5. 基于 learned-WM H2 目标的 Critic warmup。

当前正式 PPO 使用 `frozen_wm`，Actor 已从 0 更新训练到 200 次以上。固定 256 个 VAGEN 关卡上的成功率从 **13.28%** 快速提高到最高 **61.72%**，随后长期停留在约 **59%–62%**。这说明 Actor 已经学到有效策略，但目前处于明显平台期，还没有完成 1000 次 Actor 更新，也没有完成多随机种子或 `alternating_wm` 对照，因此只能判定为“有明确正向结果，正式实验尚未完成”。

## 2. 整体框架

### 2.1 World Model

World Model 的视觉输入来自原生 Qwen2.5-VL。每张图像保留为 36 个有顺序的空间 token，不对这些 token 做全局 mean pooling。这样玩家、箱子、目标和墙体的位置关系仍然保留在表示中。

V-JEPA 不参与部署时的前向输入，也没有替代 Qwen。它是一个冻结的训练期教师：离线提取的 V-JEPA 空间特征用于约束 Qwen posterior、动作条件 prior 以及动作导致的状态变化。正式推理和 PPO 阶段只依赖 Qwen World Model。

这套设计的核心是：World Model 不只追求较低的 latent loss，还必须在 held-out 数据上保留可解码的空间语义，并且根据动作预测正确的未来空间状态。

### 2.2 Reward Head

Reward Head 读取 World Model belief，预测不同 horizon 内达到成功状态的概率。Reward 是全局标量，因此这里允许使用空间 token 的汇聚；这与 Stage 1 的空间语义审计不能使用 mean pooling 并不冲突。

当前 Reward Head 是紧凑线性头，覆盖 H1–H8。训练、阈值校准和最终测试使用不同的数据划分，正式测试数据没有参与参数选择或阈值校准。

### 2.3 Critic warmup

Critic 使用独立的 Qwen LoRA 路径和 value/Q head。warmup 阶段冻结 Actor，在固定、level-disjoint 的 H1/H2 counterfactual panel 上学习不同动作的价值排序。

这里的 H2 目标来自已训练的 World Model 和 Reward Head，而不是 Sokoban 显式物理规则。它的作用是让 Critic 在 Actor PPO 开始前先具备可用的动作区分能力。

### 2.4 PPO

正式阶段仍采用 PPO 的 clipped policy update，但并不是最朴素的、只依赖真实环境完整回报的 vanilla PPO。训练同时使用 learned-WM exact-H2 advantage、Critic 更新、固定 H1/H2 replay 以及事务式 Actor Gate。

Actor 和 Critic 分别使用独立可训练的 Qwen LoRA adapter，并配有轻量策略头和价值头。当前 World Model 与 Reward Head 冻结，因此本次运行首先验证表示、奖励和价值已经固定时，Actor 是否能稳定提高真实关卡成功率。

## 3. 数据与评测协议

- World Model 原始数据：10,000 个 Sokoban episodes。
- Stage 1 语义审计：9,000 个训练 episodes，1,000 个 held-out validation episodes。
- Stage 1 spatial-v2 repair：8,500 个训练 episodes和 500 个 repair validation episodes；正式 1,000 个 validation episodes不参与 repair 训练或 checkpoint 选择。
- Reward Head：训练、validation、calibration 和 official test 相互隔离；official test 只做最终审计。
- Critic warmup：固定 H1/H2 train replay 与 held-out validation panel 分离，official evaluation 关卡不进入 Critic 训练。
- PPO evaluation：固定使用 `data/vagen_mirror_testset/sokoban_256.json` 的 256 个 VAGEN 关卡，seed 固定；每 10 次被接受的 Actor PPO update 评测一次。
- W&B 横坐标：`actor_ppo_update`，只在 Actor 更新被 Gate 接受后加一；被拒绝并回滚的候选更新不推进横坐标。
- checkpoint：每 50 次被接受的 Actor 更新覆盖保存 `latest.pt`，并按周期评测成功率保存 `best.pt`。

需要特别说明：当前周期评测集合同时参与 `best.pt` 选择。因此在严格论文协议中，这 256 个关卡应视为 validation/evaluation panel；最终方法比较最好再保留一个从未用于选 checkpoint 或调参的独立 test set。如果 VAGEN 官方协议明确允许在这 256 个关卡上反复选模，则应在论文中如实说明协议。

## 4. 各阶段实验结果

### 4.1 Stage 1：初始 World Model

初始 Stage 1 没有通过完整语义 Gate。posterior 已经能够从真实图像中恢复较好的空间信息，但 prior 无法可靠预测动作后的未来状态。

| 指标 | 结果 | 判断 |
|---|---:|---|
| posterior 玩家位置准确率 | 97.02% | 良好 |
| posterior 箱子位置准确率 | 89.14% | 良好 |
| posterior 目标位置准确率 | 91.51% | 良好 |
| posterior 墙体 macro-F1 | 91.42% | 良好 |
| posterior joint accuracy | 86.70% | 良好 |
| prior changed-state joint accuracy | 24.84% | 未通过 |
| counterfactual changed-state joint accuracy | 24.92% | 未通过 |

这说明问题不在 Qwen 看不懂当前画面，而在动作条件 prior 没有把当前空间状态正确推进到下一状态。

报告：[`stage1_wm/stage1_semantic_gate_report.json`](../checkpoints/qwen_vjepa_teacher_seed1/stage1_wm/stage1_semantic_gate_report.json)

### 4.2 Stage 1：第一次通用 prior LoRA repair

第一次 repair 保留了 posterior，但动作动力学改善不足：

- prior changed-state joint accuracy：26.06%；
- counterfactual changed-state joint accuracy：29.20%；
- 最终仍为 **FAIL**。

这次结果证明，仅增加通用 prior LoRA 训练不足以让目标直接对齐“动作后空间位置是否正确”。

报告：[`stage1_prior_repair_lora/stage1_semantic_gate_report.json`](../checkpoints/qwen_vjepa_teacher_seed1/stage1_prior_repair_lora/stage1_semantic_gate_report.json)

### 4.3 Stage 1：position-aware spatial-v2 prior repair

第二次 repair 使用位置感知的 prior，并直接围绕玩家、箱子和 counterfactual 动作后的空间变化训练。最终完整 Stage-1 semantic Gate 为 **PASS**。

| 指标 | 结果 |
|---|---:|
| posterior joint accuracy | 87.21% |
| prior changed-state joint accuracy | 88.81% |
| prior changed-state 玩家准确率 | 97.87% |
| prior changed-state 箱子准确率 | 90.61% |
| counterfactual changed-state joint accuracy | 87.46% |
| counterfactual 玩家准确率 | 97.41% |
| counterfactual 箱子准确率 | 89.60% |

这不仅超过 Gate 的最低线，而且相对第一次 repair 有大幅提升。因此 Stage 1 的最终可用 checkpoint 是：

`checkpoints/qwen_vjepa_teacher_seed1/stage1_prior_repair_spatial_v2/best.pt`

报告：[`stage1_prior_repair_spatial_v2/stage1_semantic_gate_report.json`](../checkpoints/qwen_vjepa_teacher_seed1/stage1_prior_repair_spatial_v2/stage1_semantic_gate_report.json)

### 4.4 Stage 2：Reward Head

Reward Head Gate 为 **PASS**，并且 validation 与独立 official test 的结果接近，没有出现明显只在训练/验证划分上有效的问题。

| 指标 | Validation/Calibration | Official test |
|---|---:|---:|
| AUROC | 95.44% | 95.36% |
| Average Precision | 88.19% | 88.49% |
| Brier score | 0.0757 | 0.0749 |
| Accuracy | 89.22% | 89.05% |
| Precision | 75.24% | 74.97% |
| Recall | 86.78% | 86.14% |
| ECE | 0.0240 | 0.0210 |

分 horizon AUROC 从 H1 的 99.98% 逐渐下降到 H8 的 78.17%，全部超过 65% Gate。长 horizon 明显更弱，但当前 PPO 使用 H2，因此与当前训练范围是匹配的。若以后把 imagination horizon 扩展到 H6–H8，需要重新评估长期奖励误差。

最终 checkpoint：

`checkpoints/qwen_vjepa_teacher_seed1/stage2_reward_head/best.pt`

报告：[`stage2_reward_head/reward_head_gate_report.json`](../checkpoints/qwen_vjepa_teacher_seed1/stage2_reward_head/reward_head_gate_report.json)

### 4.5 Stage 3：Critic warmup

Critic warmup 在 20 个更新后通过 Gate，Actor 在此阶段保持冻结。整体结果如下：

| 指标 | 结果 | Gate |
|---|---:|---:|
| held-out explained variance | 0.6436 | — |
| explained variance EMA | 0.5917 | ≥ 0.10 |
| MSE improvement | 63.91% | ≥ 5% |
| top-1 accuracy | 100% | ≥ 60% |
| pairwise accuracy | 95.48% | ≥ 60% |
| Q margin | 0.4679 | ≥ 0.001 |

initial/suffix 与 H1/H2 四个 bucket 全部通过各自 Gate，其中最弱的是 initial-H2 的 EV EMA 0.2317，但仍显著高于阈值。Critic 因此具备进入 PPO 的最低动作排序能力。

最终 checkpoint：

`checkpoints/qwen_vjepa_teacher_seed1/stage3_critic_warmup/best.pt`

报告：[`stage3_critic_warmup/warmup_report.json`](../checkpoints/qwen_vjepa_teacher_seed1/stage3_critic_warmup/warmup_report.json)

### 4.6 Stage 4：正式 frozen-WM PPO

第一次 PPO 启动没有真正载入 Stage 3 的 Critic checkpoint，而是在 Stage 2 权重上重新创建 Actor/Critic，导致 Actor 一直停留在 update 0，并最终因缺少 counterfactual Actor validation panel 报错。该运行只用于定位工程问题，不能作为实验结果。

修复后的 `stage4_ppo_frozen_wm_v2` 正确恢复了 Stage 3 Critic、固定 validation/replay panel 和 optimizer 状态。PPO 可以持续推进，说明此前的 checkpoint 恢复、fixed replay 拼接和事务 Gate 问题已经解决。

截至本快照，训练日志已经到 Actor update 219；最近落盘的 `latest.pt` 是每 50 步策略下的 Actor update 200。固定 VAGEN-256 evaluation 的变化为：

| Actor update | 成功数 | success rate |
|---:|---:|---:|
| 0 | 34/256 | 13.28% |
| 10 | 120/256 | 46.88% |
| 20 | 152/256 | 59.38% |
| 30 | 153/256 | 59.77% |
| 40 | 158/256 | **61.72%** |
| 50 | 152/256 | 59.38% |
| 70 | 157/256 | 61.33% |
| 90 | 158/256 | **61.72%** |
| 100–130 | 152/256 | 59.38% |
| 140 | 152/256 | 59.38% |
| 150 | 154/256 | 60.16% |
| 160–210 | 152/256 | 59.38% |

最重要的观察是：

1. **学习确实发生了。** 前 20 次 Actor 更新将成功率从 13.28% 提升到 59.38%，不可能用随机波动解释。
2. **训练已经进入平台期。** update 40 之后没有继续形成稳定上升趋势，主要在 59%–62% 之间波动。
3. **当前策略只解决了短解关卡。** update 140 时，长度 1 和 2 的关卡成功率均为 100%，长度 4 和 5 的关卡均为 0%。当前 59.38% 恰好对应所有 len-1 与 len-2 关卡之和 152/256。这说明策略学会了短程动作模式，但没有掌握需要更长规划的关卡。
4. **没有动作塌缩。** update 140 的四类动作比例约为 19.68%、17.80%、30.25% 和 32.27%，四个动作都有覆盖。
5. **事务 Gate 正常通过。** update 140–219 的四个 H1/H2 bucket 仍全部满足 top-1/pairwise ≥ 0.60，没有再次出现旧版 Gate 的“要求每一步严格单调上涨”问题。
6. **出现新的校准风险。** Actor ranking accuracy 仍高，但 validation 中部分 bucket 的 explained variance 持续转负；例如 update 218 的 initial-H2 EV 约为 -10.49，suffix-H2 EV 约为 -3.49，同时 margin 已扩大到约 1.1。也就是说 Actor 保留了正确动作顺序，却越来越不能匹配教师 Q 值的相对尺度。当前 Gate 只约束排序和防止大幅退化，不会拦截这种幅值漂移。

训练日志：[`stage4_ppo_frozen_wm_v2/train.log`](../checkpoints/qwen_vjepa_teacher_seed1/stage4_ppo_frozen_wm_v2/train.log)

## 5. 已定位并修复的工程问题

### 5.1 PPO 从错误 checkpoint 启动

旧脚本只做了 Stage 3 preflight，却没有在 fresh run 中把 Stage 3 `best.pt` 传入训练器。现在 fresh run 必须从已 release 的 Critic checkpoint 启动，并检查 H1/H2 panel 是否完整。

### 5.2 Actor update 一直为 0

原因不是 PPO 没有 backward，而是训练器首先重新执行 Critic warmup，Actor 被设计为冻结。恢复正确 Stage 3 checkpoint 后，Actor 从自己的 update 0 正常开始计数。

### 5.3 counterfactual validation panel 丢失

Stage 3 固定 panel 现在随 checkpoint 保存和恢复；正式 PPO 启动时缺少 panel 会直接在 preflight 失败，避免训练很久后才报错。

### 5.4 事务 Gate 过度严格

旧 Gate 虽然四个 bucket 均已通过绝对标准，仍要求平均 ranking score 几乎逐步单调上升，超过 0.005 的正常波动就回滚，最终连续拒绝 10 次。

现在的规则是：

- 四个 bucket 的 top-1 和 pairwise 绝对 Gate 仍为 0.60；
- 一旦全部通过，任何 bucket 重新跌破绝对 Gate 才强制回滚；
- 平均 ranking score 允许 0.05 的正常波动；
- 不再要求每一步严格单调上涨。

### 5.5 fixed replay 拼接报错

`concatenate` 原先只在一个局部分支内定义，正式 PPO 启用 fixed replay 后触发 `UnboundLocalError`。现已提升为公共 PPO batch 拼接函数，并通过相关测试。

### 5.6 checkpoint 与在线 replay 不同步

checkpoint 每 50 个 Actor update 保存一次，但在线 episode 每次更新都落盘。若在 update 218 中断，恢复时模型回到 update 200，而磁盘可能已有 218 条在线 episode。

现在通过“可见性边界”解决：恢复后的模型先只看到与 checkpoint 对齐的前 200 条 episode，随后按 Actor update 逐步复用 201–218，不删除已收集数据，也不会把未来 replay 提前泄漏给旧模型。

## 6. 当前仍需关注的问题

### 6.1 PPO 平台期是当前首要问题

成功率平台不是 evaluation 太少造成的偶然现象。它具有很清晰的结构：短解关卡全部成功，较长关卡全部失败。继续盲目增加相同 H2 更新，很可能只会强化已有短程策略，而不会自动产生 H4/H5 规划能力。

建议先让当前 run 至少完成下一个稳定 checkpoint 和 evaluation，再重点审计：

- len-4/len-5 失败轨迹是在第一步就选错，还是中途循环/超时；
- H2 imagination 是否能区分对长程成功有意义、但两步内没有即时收益的动作；
- Reward Head 的 H3–H5 信号能否安全用于更长 rollout；
- Critic/Actor 的 Q 排序正确但数值尺度持续漂移是否影响 PPO advantage。

如果长关卡确实需要超过两步的信用分配，下一步应做独立对照实验，而不是直接覆盖当前配置：例如 H2 保持为基线，新增 H4 或多 horizon 训练分支，并重新设置对应 Critic Gate。

### 6.2 只看 ranking Gate 不足以监控数值漂移

当前事务 Gate 能防止动作排序崩溃和单动作塌缩，但不能保证 Actor score 与教师 Q 的尺度一致。负 EV 与不断扩大的 margin 应加入监控，暂时不建议直接设成硬回滚 Gate，因为 Actor logit 本来不必精确回归 Q 值；应先确认这些指标与真实 success plateau 是否相关，再决定使用校准损失、margin 上限或软正则。

### 6.3 Critic Gate 证明的是“拟合 learned-WM 目标”

Critic warmup 的高 top-1 和 pairwise accuracy 证明 Critic 能拟合当前 World Model + Reward Head 产生的 H2 价值关系，但它本身不能证明这些关系在真实环境中始终正确。VAGEN 真实 rollout success 才是最终外部验证。目前从 13.28% 到约 60% 的提升提供了正证据，但长关卡失败也暴露了 H2 目标的能力边界。

### 6.4 尚未完成的实验

- frozen-WM PPO 尚未达到 1000 个 accepted Actor updates；
- `alternating_wm` 尚未在本套 Qwen + V-JEPA Teacher 实现上完成正式对照；
- 尚未完成多个 seed；
- 尚未完成独立最终 test 的一次性评测；
- 尚未形成 frozen-WM、alternating-WM、无 V-JEPA teacher 等消融对比。

## 7. 推荐的后续顺序

1. 保留当前 frozen-WM run，不覆盖其目录，继续观察到至少 Actor update 250/300，并记录 len-4/len-5 的失败类型。
2. 不因排名 EV 变负立即回滚；先增加只读诊断，比较真实成功/失败动作与 learned H2 排序的一致率。
3. 如果长关卡仍为 0%，新建独立 H4/多 horizon 分支，不修改当前 H2 基线。
4. frozen-WM 配置稳定后，再建立 `alternating_wm` 分支，并沿用同一 VAGEN evaluation、seed、Actor/Critic 结构和 Gate，只改变 WM 是否周期更新。
5. 最终至少运行多个 seed，报告均值与方差；checkpoint 选择只使用 validation panel，最后一次在 untouched test 上评测。

## 8. 关键文件索引

- 架构说明：[`qwen_vjepa_teacher_world_model.md`](qwen_vjepa_teacher_world_model.md)
- Stage 1 最终 checkpoint：`checkpoints/qwen_vjepa_teacher_seed1/stage1_prior_repair_spatial_v2/best.pt`
- Stage 2 Reward Head：`checkpoints/qwen_vjepa_teacher_seed1/stage2_reward_head/best.pt`
- Stage 3 Critic：`checkpoints/qwen_vjepa_teacher_seed1/stage3_critic_warmup/best.pt`
- Stage 4 PPO latest：`checkpoints/qwen_vjepa_teacher_seed1/stage4_ppo_frozen_wm_v2/latest.pt`
- Stage 4 PPO best：`checkpoints/qwen_vjepa_teacher_seed1/stage4_ppo_frozen_wm_v2/best.pt`
- 正式 PPO 脚本：[`scripts/run_qwen_vjepa_ppo.sh`](../scripts/run_qwen_vjepa_ppo.sh)

恢复当前 PPO 的命令为：

```bash
cd MBRL0901
GPU_ID=0 SEED=1 RESUME=1 bash scripts/run_qwen_vjepa_ppo.sh
```

恢复时会从最近一次 `latest.pt` 开始，而不是从日志中最后一个尚未落盘的 Actor update 开始。当前每 50 次 accepted Actor update 保存一次，这是存储开销与中断重算成本之间的折中。
