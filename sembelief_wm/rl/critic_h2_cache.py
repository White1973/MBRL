
    +            seen.add(id(parameter))
    +    return result

• Ran left=/personal/jiayu2026/code/code_new/MBRL/sembelief_wm
  │ right=/personal/jiayu2026/code/MBRL0901/sembelief_wm
  │ for f in rl/trajectory.py train/losses.py train/targets.py train/windowing.py trainers/wm_refresher.py; do echo "### $f"; git diff --no-index --unified=1 --
  │ … +1 lines
  └ diff --git a/personal/jiayu2026/code/code_new/MBRL/sembelief_wm/train/losses.py b/personal/jiayu2026/code/MBRL0901/sembelief_wm/train/losses.py
    index ae096c2..0657391 100644
    … +610 lines (ctrl + t to view transcript)
    +            diagnostics=diagnostics,
             )

• Explored
  └ Search open_loop|delta_cosine|inverse_action|observation_anchor|vjepa|validation_guard|grouped_gradient|optimizer|reward_positive|checkpoint in trainer.py
    Read trainer.py
    Read trainer.py

• Ran python - <<'PY'
  │ import ast
  │ from pathlib import Path
  │ … +14 lines
  └ data/tokenizers/qwen.py    Native Qwen2.5-VL visual tokenization for the world model.    QwenVisionTokenizer, QwenVJEPAObservationTokenizer
    diagnostics/frozenlake_h1_h2_repair.py    Isolated FrozenLake prior repair with explicit H1 and open-loop H2 supervision.    H1H2LossWeights, run_h1_h2_repair
    … +50 lines (ctrl + t to view transcript)
    rl/transition_target_causal_diagnostic.py    Causal 2x2 audit of Reward history and World-Model next-state errors.    run_transition_target_causal_diagnostic
    train/ranking_reward.py    Solver-supervised H3 ranking repair for the compact terminal Reward Head.    train_ranking_reward_head

• Ran python - <<'PY'
  │ from pathlib import Path
  │ L=Path('/personal/jiayu2026/code/code_new/MBRL/sembelief_wm'); R=Path('/personal/jiayu2026/code/MBRL0901/sembelief_wm')
  │ … +10 lines
  └ prod 53
    train/ranking_reward.py
    … +77 lines (ctrl + t to view transcript)
    init 1
    lewm/__init__.py

• Ran python - <<'PY'
  │ from pathlib import Path
  │ L=Path('/personal/jiayu2026/code/code_new/MBRL/sembelief_wm'); R=Path('/personal/jiayu2026/code/MBRL0901/sembelief_wm')
  │ … +9 lines
  └   990 diagnostics/sokoban_prior_repair_v2.py
      839 diagnostics/frozenlake_lora_spatial_repair.py
    … +49 lines (ctrl + t to view transcript)
       51 rl/real_return_anchor.py
       41 model/checkpoint_semantics.py

• Ran left=/personal/jiayu2026/code/code_new/MBRL/sembelief_wm
  │ right=/personal/jiayu2026/code/MBRL0901/sembelief_wm
  │ find "$left" -type f -name '*.py' -not -path '*/__pycache__/*' -printf '%P\n' | sort | comm -12 - <(find "$right" -type f -name '*.py' -not -path '*/__pycache__/*'
  │ … +2 lines
  └ added 13760 deleted 600

• Ran python - <<'PY'
  │ from pathlib import Path
  │ L=Path('/personal/jiayu2026/code/code_new/MBRL/sembelief_wm'); R=Path('/personal/jiayu2026/code/MBRL0901/sembelief_wm')
  │ … +10 lines
  └ 53 16568 24 1912 1

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 已优先完成 sembelief_wm/ 内的源码对比。结论是：MBRL0901 基本是 code_new/MBRL 的大幅扩展版，不是少量参数调整。

  对比范围为所有 .py 文件，排除了 __pycache__：

   类别                            数量
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━
   code_new/MBRL 中 Python 文件      97
  ──────────────────────────────  ──────
   MBRL0901 中 Python 文件          175
  ──────────────────────────────  ──────
   同路径但内容不同                  40
  ──────────────────────────────  ──────
   其中生产/运行代码                 33
  ──────────────────────────────  ──────
   其中测试代码                       7
  ──────────────────────────────  ──────
   仅 MBRL0901 存在                  78
  ──────────────────────────────  ──────
   仅 code_new/MBRL 存在              0
  ──────────────────────────────  ──────
   双方完全一致                      57

  40 个同路径差异文件合计约增加 13,760 行、删除 600 行；此外 MBRL0901 还有约 16,568 行全新的生产代码。

  ## 一、同路径但实现不同的生产代码

  ### 1. Agent 与配置

  - agent/config.py
      - PPO 新增 target_kl
      - 新增终点成功型 reward mapping、reward 缩放和置信度阈值
      - WM refresh 新增 reward loss、冻结 reward head、验证 batch 等配置

  - agent/phase2.py
      - device 转移增加 episode_success
      - 增加 V-JEPA semantic teacher token/mask

  - agent/real_env_ppo.py
      - GAE 区分真正终止和时间截断
      - 截断轨迹允许 value bootstrap
      - 成功终止保持零 bootstrap

  ### 2. 数据采集与 imagined rollout

  - collectors/adapters.py
      - 新增 action_id_offset，兼容 Sokoban 老 checkpoint 使用的 1–4 action ID

  - collectors/belief_sampler.py
      - 起始 belief 必须满足剩余真实轨迹长度不少于 imagined horizon
      - 新增 belief 与日志 action 联合采样，供 offline BC 使用
      - grounding 显式 no_grad
      - 支持提高早期状态采样概率

  - code/MBRL0901/sembelief_wm/collectors/imagined.py
      - 增加 fixed_horizon 与 predicted_success 两种终止语义
      - 支持强制第一步 action/完整 action sequence
      - 支持 relative action-value reward shaping
      - 增加真实的 done、padding mask 和 bootstrap 处理
      - 记录 base reward、shaping reward、reward logit 和动作排名诊断

  - code/MBRL0901/sembelief_wm/collectors/real.py
      - 支持固定 seeds、固定 Sokoban layouts
      - 支持 epsilon exploration 和强制 action
      - 支持收集完整 policy latent trajectory
      - 收集 V-JEPA teacher features
      - action ID 增加 WM offset
      - 新增 RealEnvEvaluator
      - 增加动作塌缩、初始 logits 漂移等在线评测指标

  - collectors/reward_transforms.py
      - 新增：
          - terminal_success
          - terminal_success_scaled
          - terminal_success_conservative
          - per_transition_success_conservative

      - reward mapping 从简单 sigmoid/logit 转为可表达稀疏终点奖励和置信度门控

  ### 3. 数据格式、环境与 replay

  - data/adapters/frozenlake.py
      - 从内部离散状态环境改成 Gymnasium RGB 环境
      - 使用 VAGEN 随机地图生成器
      - observation spec 从 discrete_state 改为 rgb

  - data/adapters/sokoban.py
      - 可强制要求真实 gym_sokoban，不再静默降级 mock
      - 同时固定 Python/NumPy 随机种子
      - 支持按最短解长度筛选关卡
      - 新增 reset_with_room
      - 成功判断改为“所有箱子位于目标点”，不依赖 reward 大小

  - data/datasource.py
      - batch 新增 episode success label
      - padding/collate V-JEPA teacher token 及 mask

  - code/MBRL0901/sembelief_wm/data/replay.py
      - MixedDataSampler 对小比例 online 数据采用无偏随机取整
      - 增加采样来源统计
      - 新增 UnifiedRandomReplayPool
      - replay 可按照 Actor checkpoint 控制 online episode 可见范围

  - data/schema.py
      - TokenizedEpisode 新增 semantic_teacher_tokens
      - 序列化/反序列化同步支持该字段

  - data/storage.py
      - manifest 增加 semantic teacher shape

  - data/tokenizers/image.py
      - V-JEPA tokenizer 增加 teacher-only feature 提取接口

  - data/tokenizers/__init__.py
      - 导出新的 Qwen tokenizer；属于 API/import 行为变化

  ### 4. 世界模型和 backbone

  - code/MBRL0901/sembelief_wm/ema.py
      - 差异非常大：code_new/MBRL 中该文件实际上重复定义了 BeliefState/SequenceBatch 等类型
      - MBRL0901 中替换为真正的 EMATeacher 和 ema_variance_loss
      - 这不是普通增量修改，而是整个文件职责发生变化

  - code/MBRL0901/sembelief_wm/model/backbone_qwen.py
      - 支持同一个 Qwen base 上的多个独立 LoRA adapter
      - WM、Actor、Critic adapter 可独立训练
      - 修复 bidirectional attention：显式传入 4D 非因果 mask
      - bidirectional 模式自动避开不兼容的 FlashAttention2
      - 增加 adapter 参数隔离和 trainability 控制

  - model/policy_backbone.py
      - 新增 from_shared_adapter
      - policy 可以共享 Qwen base、但使用独立 LoRA
      - trainable_parameters() 只返回指定 adapter 参数

  - model/reward.py
      - reward head 支持 legacy full MLP、linear head、compact MLP
      - 新增 forward_pooled，支持缓存 pooled features 后单独训练 reward classifier

  - code/MBRL0901/sembelief_wm/model/transition.py
      - token 顺序改为 observation → action → belief
      - 支持 posterior action-free
      - 支持 visual_anchor grounding
      - 支持 posterior observation residual
      - prior 可使用独立 LoRA、低秩 residual 或 state-action PCA adapter
      - 新增 PriorResidualAdapter、PriorStateActionAdapter

  - code/MBRL0901/sembelief_wm/model/world_model.py
      - 增加真实 open-loop rollout loss
      - 增加 detached prior reward supervision
      - 增加 action-delta cosine 和 inverse-action loss
      - 增加 observation anchor
      - 增加 V-JEPA prior/posterior/delta teacher loss
      - reward label 支持 success threshold
      - 增加正确对齐的 terminal reward auxiliary loss

  - model/visual_encoder.py
      - 只有文档说明变化，没有可执行逻辑变化；明确其已变为 legacy V-JEPA input 路径

  ### 5. Pipeline 编排

  - code/MBRL0901/sembelief_wm/pipelines/assemble.py
      - 严格检查 PPO optimizer 与 WM 参数不重叠
      - Actor、Critic、WM 使用独立 LoRA adapter
      - 接入 unified replay、offline BC、behavior rehearsal
      - 接入新的 imagined termination/reward mapping
      - 支持 Le-WM relative H3 shaping
      - 对短于 rollout horizon 的 episode 做过滤和重采样
      - 固定 policy forward mode，避免 dropout 造成伪 PPO ratio

  - code/MBRL0901/sembelief_wm/pipelines/mbrl_train.py
      - 是差异最大的文件，约增加 4,576 行
      - 从简单的“采样→PPO→评估”变成带状态机和安全门控的训练管线
      - 新增 critic warmup、固定 validation panel、real-return critic anchor
      - 新增 counterfactual H2/Q grounding
      - 新增 Actor/Critic gate、失败回滚和 optimizer transaction
      - 新增 joint critic、target critic、PCGrad
      - checkpoint 按 Actor update 对齐，并支持 best checkpoint
      - replay 可见性与 checkpoint 恢复绑定

  ### 6. PPO 和 Actor/Critic

  - code/MBRL0901/sembelief_wm/rl/llm_policy.py
      - Actor 支持 qwen_pooled、qwen_slotwise、latent_belief、frozen_vlm
      - Critic 支持 pooled V、ordered V、slotwise Q、latent critic
      - Actor/Critic 可用不同 LoRA adapter
      - 新增历史 behavior policy snapshot 和 behavior KL
      - 新增 expert rehearsal 接口
      - Q critic bootstrap 改为对四个 action 求精确策略期望
      - 增加 batch-global action-gradient projection，抑制全局动作塌缩

  - code/MBRL0901/sembelief_wm/rl/ppo.py
      - Actor/Critic optimizer 参数组和学习率分离
      - 支持 actor-only、critic-only 更新
      - 增加 target-KL early stop
      - 增加 behavior KL、behavior BC rehearsal、entropy floor
      - PPO 前重新计算 old log-prob，避免 Qwen/bf16 batch-shape 数值差异
      - 支持完整四动作 group 的 Critic 更新
      - 新增 imagined H2 + real return 联合 Critic 和 PCGrad
      - 支持 Actor 或完整 PPO transaction snapshot/rollback

  - rl/trajectory.py
      - trajectory 新增 reward/shaping/ranking 诊断字段

  ### 7. Phase-1/WM 训练

  - train/losses.py
      - 新增 open-loop dynamics、prior reward loss
      - SIGReg 支持 mean/n-scaled 两种尺度
      - reward target 支持 threshold
      - 新增正确对齐的 terminal reward BCE

  - train/targets.py
      - reward 二值标签从固定 reward > 0 改为可配置 threshold

  - train/windowing.py
      - 正确计算 terminal window mask
      - 在 BPTT window 中传递当前和前一帧 teacher features

  - code/MBRL0901/sembelief_wm/train/trainer.py
      - 接入 open-loop、action、observation、V-JEPA 辅助目标
      - 增加不同模块的梯度范数诊断
      - optimizer 支持 inverse-action 独立参数组
      - 新增 held-out validation guard
      - 改进 checkpoint/latest checkpoint 管理

  - trainers/wm_refresher.py
      - Phase-2 refresh 使用独立的 Phase-1 config view
      - 不再用 requires_grad_(True) 粗暴解冻整个 Qwen
      - 精确恢复 WM 与 PPO adapter 的参数所有权
      - 增加固定验证集和详细 reward/dynamics diagnostics

  ## 二、仅 MBRL0901 存在的生产代码

  共 53 个。

  ### 原生 Qwen、checkpoint、Reward

  - data/tokenizers/qwen.py
  - model/checkpoint_semantics.py
  - train/ranking_reward.py

  其中 qwen.py 实现原生 Qwen2.5-VL image feature 提取和固定空间网格重采样；checkpoint_semantics.py 会拒绝使用 bidirectional-attention 修复前的旧 checkpoint。

  ### 新增 diagnostics：7 个

  - diagnostics/frozenlake_h1_h2_repair.py
  - diagnostics/frozenlake_h2_audit.py
  - diagnostics/frozenlake_lora_spatial_repair.py
  - diagnostics/frozenlake_prior_repair.py
  - diagnostics/frozenlake_spatial.py
  - diagnostics/sokoban_prior_repair_v2.py
  - diagnostics/sokoban_stage1_audit.py

  主要覆盖 FrozenLake/Sokoban 的空间解码、H1/H2 prior、action dynamics 和独立 LoRA repair。

  ### 独立 LEWM 子系统：11 个

  - lewm/action_grounding_audit.py
  - lewm/actor_probe.py
  - lewm/config.py
  - lewm/core_contract.py
  - lewm/data.py
  - lewm/entrypoint.py
  - lewm/final_levels.py
  - lewm/grounding_audit.py
  - lewm/intermediate_return_audit.py
  - lewm/pipeline.py
  - lewm/validation.py

  这不是普通工具集合，而是一套独立的 Le-WM 状态机、数据契约、验证和 Actor probe 流程。

  ### 新增 RL/诊断实现：32 个

  - rl/action_q_probe.py
  - rl/counterfactual_action_audit.py
  - rl/counterfactual_h2_ppo.py
  - rl/counterfactual_spatial_decoder_repair.py
  - rl/critic_h2_cache.py
  - rl/critic_warmup_probe.py
  - rl/fixed_counterfactual_q_probe.py
  - rl/fixed_scalar_v_probe.py
  - rl/formal_q_pretrain.py
  - rl/frozen_actor_critic_cross_diagnostic.py
  - rl/frozen_posterior_spatial_probe.py
  - rl/full_trajectory_h8.py
  - rl/imagined_real_action_ranking_audit.py
  - rl/initial_h12_posterior_cache.py
  - rl/joint_critic_causal_diagnostic.py
  - rl/long_horizon_critic_grounding.py
  - rl/near_terminal_actor_warmup.py
  - rl/near_terminal_critic_repair.py
  - rl/near_terminal_ranking_audit.py
  - rl/offline_bc.py
  - rl/online_actor_warmup.py
  - rl/online_counterfactual_256_audit.py
  - rl/posterior_target_bucket_diagnostic.py
  - rl/prior_reward_decomposition_audit.py
  - rl/real_return_anchor.py
  - rl/runtime_matched_h2_sequence_cache.py
  - rl/runtime_matched_spatial_decoder.py
  - rl/runtime_matched_state_action_prior_repair.py
  - rl/spatial_decoder_real_posterior_audit.py
  - rl/symbolic_sokoban.py
  - rl/terminal_reward_coordinate_audit.py
  - rl/transition_target_causal_diagnostic.py

  核心新增能力是：counterfactual 四动作 Q、real-return grounding、固定 H1/H2/H8 cache、Actor/Critic warmup、空间解码 repair，以及 imagined/real ranking 对齐审计。

  ## 三、测试文件差异

  同路径修改的 7 个测试文件：

  - collectors/tests/test_belief_sampler.py
  - collectors/tests/test_imagined.py
  - collectors/tests/test_real.py
  - pipelines/tests/test_assemble_direct.py
  - pipelines/tests/test_assemble.py
  - pipelines/tests/test_pipeline.py
  - trainers/tests/test_wm_refresher.py

  MBRL0901 另新增 24 个测试文件，分布为：

  - lewm/tests/：4 个
  - diagnostics/tests/：2 个
  - model/tests/：1 个
  - data/tests/：1 个
  - rl/tests/：16 个

  ## 四、最值得优先审查的文件

  如果目的是理解“MBRL0901 的算法到底与原 MBRL 有什么不同”，建议依次看：

  1. code/MBRL0901/sembelief_wm/pipelines/mbrl_train.py
  2. code/MBRL0901/sembelief_wm/rl/ppo.py
  3. code/MBRL0901/sembelief_wm/rl/llm_policy.py
  4. code/MBRL0901/sembelief_wm/pipelines/assemble.py
  5. code/MBRL0901/sembelief_wm/model/world_model.py
  6. code/MBRL0901/sembelief_wm/model/transition.py
  7. code/MBRL0901/sembelief_wm/model/backbone_qwen.py
  8. code/MBRL0901/sembelief_wm/train/trainer.py

  整体上，MBRL0901 的主要方向是：原生 Qwen 视觉表示、V-JEPA teacher、严格分离的 WM/Actor/Critic LoRA、action-aware prior、真实/想象联合 Critic、训练质量门控和可回滚
  PPO。外层 scripts/ 和根目录入口尚未包含在本轮重点分析中。

─ Worked for 6m 17s ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


› 帮我分析对于### 新增 diagnostics：7 个和新增 RL/诊断实现：32 个哪些是在训练框架中保留，成为训练过程中使用的，哪些后来被弃用了在最新的训练中对于sokoban和frozenlake任
  务


• 我会按“是否被核心训练管线直接导入”“是否仍由最新 Sokoban/FrozenLake 启动脚本调用”“只用于离线审计/修复”“已明确移除或没有现行入口”四层判断，避免仅凭文件仍存在就认定仍在
  使用。

• Ran python - <<'PY'
  │ import ast
  │ from pathlib import Path
  │ … +41 lines
  └ sembelief_wm.diagnostics.frozenlake_h1_h2_repair    scripts/frozenlake/repair_stage1_h1_h2.py
    sembelief_wm.diagnostics.frozenlake_h2_audit    scripts/frozenlake/audit_stage1_h2_open_loop.py,sembelief_wm/diagnostics/frozenlake_h1_h2_repair.py
    … +49 lines (ctrl + t to view transcript)
    sembelief_wm.rl.transition_target_causal_diagnostic    scripts/train_mbrl.py,sembelief_wm/rl/frozen_posterior_spatial_probe.py,sembelief_wm/rl/
    spatial_decoder_real_posterior_audit.py

• Explored
  └ Search *.sh in scripts
    List docs
    Read REMOVED_EXPERIMENTS.md
    Read MIGRATION.md
    Read qwen_vjepa_experiment_analysis_20260903.md, qwen_vjepa_teacher_world_model.md, run_qwen_vjepa_ppo.sh, run_qwen_vjepa_ppo_alternating_wm.sh,
         run_frozenlake_training.sh, run_frozenlake_prior_spatial_repair.sh, run_qwen_vjepa_spatial_prior_repair.py
    Search from sembelief_wm\.(diagnostics|rl)\.|frozenlake_spatial_audit|critic_h2|counterfactual|near_terminal|runtime_matched|full_trajectory|joint_critic|
    online_actor|
           prior_reward|transition_target|frozen_actor|symbolic|offline_bc|long_horizon in train_mbrl.py
    Read train_mbrl.py, run_qwen_vjepa_teacher_wm.sh, run_qwen_vjepa_prior_repair_v2.sh, run_qwen_vjepa_critic_warmup.sh, build_qwen_vjepa_critic_h2_cache.py,
         preflight_qwen_vjepa_ppo.py
    Search counterfactual_h2_ppo_batch|collection_to_ppo_batch|OfflineBehaviorCloner|offline_behavior|critic_h2_cache|fixed_counterfactual|online_actor_warmup|
           action_q_probe|formal_q_pretrain|near_terminal|symbolic|long_horizon|joint_critic|runtime_matched|full_trajectory|initial_h12|prior_reward_decomposition|
           transition_target in mbrl_train.py
    Read mbrl_train.py
    Search abandon|abandoned|deprecated|legacy|retired|supersed|remove|disabled|do not|不再|弃用|废弃|移除|失败|fail|最终|formal|current|latest in docs
    Search *.pt in frozenlake
    Search *.pt in qwen_vjepa_teacher_seed1
    Read train.log, repair_report.json, stage1_semantic_gate_report.json
    Search sembelief_wm\.rl\. in train_qwen_vjepa_reward_head.py
    Search sembelief_wm\.diagnostics\. in train_qwen_vjepa_reward_head.py
    Read frozenlake_lora_spatial_repair.py
    Read frozenlake_spatial.py
    Read sokoban_prior_repair_v2.py
    Search frozenlake_prior_repair|from \.frozenlake_prior_repair in frozenlake_lora_spatial_repair.py

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 目前证据已经比较清楚：Sokoban 的正式链路已经运行到 Stage 4；FrozenLake 最新链路在 9 月 5 日完成了 Stage-1 spatial-v3，Stage 2–4 已写入统一 runner，但从现有
  checkpoint 看尚未实际跑到 Critic/PPO。因此我会把“当前已实际使用”和“最新 runner 计划使用”分开标注。

• Explored
  └ Search ^run_(base_wm|prior|spatial_prior|wm_audit|reward|critic|ppo)\(\)|run_frozenlake_prior_spatial_repair|frozenlake-spatial-audit|run_qwen_vjepa_critic_warmup|
           run_qwen_vjepa_ppo in run_frozenlake_training.sh
    Search COUNTERFACTUAL_H2_PPO|IMAGINED_CRITIC_UPDATE|offline-bc-steps|behavior-kl-coef|SYMBOLIC_PHYSICS_TERMINAL|CONTINUE_SYMBOLIC_H2_RUN|critic-h2-cache|
           CRITIC_PRETRAIN_ONLY|FORMAL_UNIFIED_PPO in run_qwen_vjepa_ppo.sh
    Search from \.\.rl\.counterfactual_h2_ppo|from \.\.rl\.offline_bc|real_return_anchor|from sembelief_wm\.diagnostics\.frozenlake_spatial in mbrl_train.py
    Search stage1_prior_repair_spatial_v2|正式 PPO 使用|Stage 4|offline BC|symbolic in qwen_vjepa_experiment_analysis_20260903.md

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 结论先说：

  - 7 个 diagnostics 文件中，5 个仍属于最新 Sokoban/FrozenLake 流程，2 个已被 FrozenLake 新 spatial-v3 路径取代。
  - 32 个新增 RL 文件中，真正进入最新正式训练主链的只有 2 个：
      - rl/counterfactual_h2_ppo.py
      - rl/critic_h2_cache.py

  - 其余 30 个里：
      - 16 个是保留的只读审计/探针，不参与正式参数更新；
      - 14 个属于旧实验、旧 repair/warmup 路径或当前关闭的可选分支。

  - “文件还在”不等于“最新训练仍使用”。很多模块只是被 train_mbrl.py 作为条件分支保留。

  判断依据是最新启动脚本、实际 checkpoint/log、Python import 调用关系，而不是文件时间或名称。

  ## 一、最新正式训练链路

  ### Sokoban 当前链路

   阶段                     当前使用的新增模块
  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Stage 1 WM               通用 train_mbrl.py
  ───────────────────────  ────────────────────────────────────────────────────
   Stage 1 prior repair     diagnostics/sokoban_prior_repair_v2.py
  ───────────────────────  ────────────────────────────────────────────────────
   Stage 1 gate             diagnostics/sokoban_stage1_audit.py
  ───────────────────────  ────────────────────────────────────────────────────
   Stage 2 Reward Head      不直接使用这 39 个模块
  ───────────────────────  ────────────────────────────────────────────────────
   Stage 3 Critic warmup    rl/critic_h2_cache.py、rl/counterfactual_h2_ppo.py
  ───────────────────────  ────────────────────────────────────────────────────
   Stage 4 PPO              rl/counterfactual_h2_ppo.py

  最新 Sokoban 正式 PPO 明确启用了：

  - COUNTERFACTUAL_H2_PPO=1
  - IMAGINED_CRITIC_UPDATE=1
  - learned-WM exact-H2 target

  见 code/MBRL0901/scripts/run_qwen_vjepa_ppo.sh:121。

  Sokoban 已实际跑到 Stage 4，最新实验说明中记录 frozen-WM PPO 已超过 200 个 Actor update，最高约 61.72%，见 code/MBRL0901/docs/experiments/
  qwen_vjepa_experiment_analysis_20260903.md:18。

  ### FrozenLake 当前链路

   阶段                            当前使用/计划使用的新增模块
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Stage 1 WM                      通用 train_mbrl.py
  ──────────────────────────────  ─────────────────────────────────────────────────────────────
   Stage 1 generic prior repair    通用 train_mbrl.py
  ──────────────────────────────  ─────────────────────────────────────────────────────────────
   Stage 1 spatial-v3 repair       diagnostics/frozenlake_lora_spatial_repair.py
  ──────────────────────────────  ─────────────────────────────────────────────────────────────
   Stage 1 final gate              diagnostics/frozenlake_spatial.py
  ──────────────────────────────  ─────────────────────────────────────────────────────────────
   Stage 2 Reward Head             不直接使用这 39 个模块
  ──────────────────────────────  ─────────────────────────────────────────────────────────────
   Stage 3 Critic warmup           计划使用 rl/critic_h2_cache.py、rl/counterfactual_h2_ppo.py
  ──────────────────────────────  ─────────────────────────────────────────────────────────────
   Stage 4 PPO                     计划使用 rl/counterfactual_h2_ppo.py

  完整编排见 code/MBRL0901/scripts/run_frozenlake_training.sh:134。

  需要区分“已经运行”和“runner 已定义”：

  - FrozenLake 已实际完成 Stage-1 spatial-v3，并通过 gate；
  - 当前 checkpoint 目录尚未看到 Stage 2 Reward、Stage 3 Critic 或 Stage 4 PPO 产物；
  - 因此 critic_h2_cache.py 和 counterfactual_h2_ppo.py 对 FrozenLake 是最新正式方案的一部分，但截至当前目录状态尚未实际跑到。

  FrozenLake spatial-v3 的最终报告在 code/MBRL0901/checkpoints/frozenlake/qwen_vjepa_teacher_seed11/stage1_prior_spatial_repair_v3/repair_report.json。

  ## 二、7 个 diagnostics 文件的状态

  ### A. 当前 Sokoban 正式链路使用

  #### diagnostics/sokoban_prior_repair_v2.py

  状态：当前使用，Stage-1 训练阶段。

  作用：

  - position-aware prior repair
  - 只更新独立 wm_prior LoRA
  - 使用玩家/箱子位置、counterfactual successor、changed-slot V-JEPA loss
  - 生成 Sokoban 当前正式 Stage-1 prior checkpoint

  入口：

  - code/MBRL0901/scripts/run_qwen_vjepa_prior_repair_v2.py:16
  - scripts/run_qwen_vjepa_prior_repair_v2.sh

  这是当前 Sokoban pipeline 中真正修改模型参数的 diagnostics 模块。

  #### diagnostics/sokoban_stage1_audit.py

  状态：当前使用，Stage-1 gate；同时被 prior repair 复用。

  作用：

  - 训练/评估 slotwise Sokoban spatial decoder
  - 验证 posterior、prior、counterfactual 后的玩家/箱子/目标/墙体信息
  - prior repair 会复用其中的状态解析、successor 和 decoder 逻辑
  - 最终 audit 会重新拟合独立 decoder，避免 repair 内 teacher 与正式验收耦合

  入口：

  - code/MBRL0901/scripts/audit_qwen_vjepa_teacher_wm.py:16
  - sokoban_prior_repair_v2.py

  它不直接作为 PPO loss，但决定 Stage-1 checkpoint 是否可以进入 Reward/Critic/PPO 阶段。

  ### B. 当前 FrozenLake 正式链路使用

  #### diagnostics/frozenlake_lora_spatial_repair.py

  状态：当前使用，Stage-1 spatial-v3 参数训练。

  作用：

  - 冻结整个 WM，只打开 wm_prior LoRA
  - 用 FrozenLake 真值位置监督实际和 counterfactual successor
  - 保持 posterior latent geometry
  - 加入 V-JEPA prior/delta 监督
  - 按 held-out spatial gate 选最佳 adapter/checkpoint

  入口：

  - code/MBRL0901/scripts/frozenlake/run_qwen_vjepa_spatial_prior_repair.py:16
  - code/MBRL0901/scripts/run_frozenlake_prior_spatial_repair.sh

  这是 FrozenLake 当前最新且已经实际执行的 repair 实现。

  #### diagnostics/frozenlake_spatial.py

  状态：当前使用，Stage-1 最终只读验收。

  作用：

  - 从 held-out episode 提取 posterior/prior/counterfactual belief
  - 拟合独立 spatial probe
  - 检查 posterior、prior、counterfactual accuracy
  - 不修改正式 WM 参数
  - gate 失败时阻止进入 Reward/Critic/PPO

  由 train_mbrl.py --frozenlake-spatial-audit 调用，见 code/MBRL0901/scripts/train_mbrl.py:1499。

  #### diagnostics/frozenlake_prior_repair.py

  状态：仍被当前 FrozenLake spatial-v3 间接使用，但不再作为独立主 repair 入口。

  当前 frozenlake_lora_spatial_repair.py 从中导入 _posterior_rows。因此：

  - 旧的 run_prior_repair() 主流程已经不是当前选择；
  - 其中的数据提取工具仍是最新实现的运行依赖；
  - 不能直接删除整个文件，否则最新 spatial-v3 会 import 失败。

  更准确的定位是：旧训练方案被替代，但部分公共实现仍在服役。

  ### C. 已被最新 FrozenLake 路径取代

  #### diagnostics/frozenlake_h1_h2_repair.py

  状态：历史实验路径，最新训练不再使用。

  历史入口：

  - scripts/frozenlake/repair_stage1_h1_h2.py
  - scripts/frozenlake/run_stage1_h1_h2_repair.sh

  现有产物停留在 stage1_h1_h2_repair_v2_smoke；最新 FrozenLake 已改用 stage1_prior_spatial_repair_v3。

  #### diagnostics/frozenlake_h2_audit.py

  状态：历史 H2 open-loop audit，最新正式 gate 不再使用。

  历史入口：

  - scripts/frozenlake/audit_stage1_h2_open_loop.py
  - scripts/frozenlake/run_stage1_h2_open_loop_audit.sh

  最新最终 gate 已统一到 frozenlake_spatial.py，repair 则统一到 frozenlake_lora_spatial_repair.py。

  ### diagnostics 最终分类

   文件                                 Sokoban                     FrozenLake                当前判断
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━
   sokoban_prior_repair_v2.py           Stage-1 训练                不使用                    保留、当前使用
  ───────────────────────────────────  ──────────────────────────  ────────────────────────  ────────────────
   sokoban_stage1_audit.py              Stage-1 gate/repair 依赖    不使用                    保留、当前使用
  ───────────────────────────────────  ──────────────────────────  ────────────────────────  ────────────────
   frozenlake_lora_spatial_repair.py    不使用                      Stage-1 训练              保留、当前使用
  ───────────────────────────────────  ──────────────────────────  ────────────────────────  ────────────────
   frozenlake_spatial.py                不使用                      Stage-1 gate              保留、当前使用
  ───────────────────────────────────  ──────────────────────────  ────────────────────────  ────────────────
   frozenlake_prior_repair.py           不使用                      当前 repair 的工具依赖    部分保留
  ───────────────────────────────────  ──────────────────────────  ────────────────────────  ────────────────
   frozenlake_h1_h2_repair.py           不使用                      旧 smoke                  已被取代
  ───────────────────────────────────  ──────────────────────────  ────────────────────────  ────────────────
   frozenlake_h2_audit.py               不使用                      旧 H2 audit               已被取代

  ## 三、32 个 RL 文件中，最新正式训练真正使用的模块

  ### 1. rl/counterfactual_h2_ppo.py

  状态：Sokoban 当前使用；FrozenLake 最新方案计划使用。

  这是当前最核心的新增 RL 文件。

  作用：

  - 对每个相同起始 belief 枚举全部 16 条 (a0, a1) 两步 action sequence
  - 生成 exact finite-H2 learned-WM target
  - 将 16 条序列聚合成 4 个第一动作的 Q target
  - Actor advantage 使用同状态四动作中心化结果
  - Critic 学动作条件 Q(s,a)，而不是把所有动作压到一个 scalar V target

  它在核心管线被直接导入，见 code/MBRL0901/sembelief_wm/pipelines/mbrl_train.py:29。

  Stage 3 和 Stage 4 runner 均明确设置 COUNTERFACTUAL_H2_PPO=1。

  ### 2. rl/critic_h2_cache.py

  状态：Sokoban Stage 3 当前使用；FrozenLake Stage 3 最新方案计划使用。

  作用：

  - 安装固定、split-safe 的 H1/H2 posterior panel
  - 保存 initial-H1、initial-H2、suffix-H1、suffix-H2 四个 bucket
  - 为 Critic warmup 和 Actor transaction gate 提供不可变 validation panel
  - checkpoint 恢复时继续保持相同的目标和验证集

  Stage 3 runner 通过 --critic-h2-cache 使用，见 code/MBRL0901/scripts/run_qwen_vjepa_critic_warmup.sh:143。

  因此，这两个文件应视为最新 Sokoban/FrozenLake 统一训练框架的正式组成部分。

  ## 四、保留在框架中，但最新配置关闭的训练功能

  ### rl/offline_bc.py

  状态：代码级保留，但最新 Sokoban/FrozenLake 正式训练均关闭。

  它仍被 pipelines/assemble.py 顶层导入，因此文件不能直接删除，否则当前 pipeline 连 import 都会失败：

  - code/MBRL0901/sembelief_wm/pipelines/assemble.py:27

  但最新 runner 明确设置：

  --offline-bc-steps 0
  --behavior-kl-coef 0
  --behavior-bc-coef 0

  见：

  - code/MBRL0901/scripts/run_qwen_vjepa_ppo.sh:244
  - code/MBRL0901/scripts/run_qwen_vjepa_critic_warmup.sh:181

  所以它属于“框架兼容功能仍在，但最新正式方法已弃用”。

  ### rl/real_return_anchor.py

  状态：保留为可选旧分支，最新 formal exact-H2 流程未启用。

  只有设置以下环境变量时才会进入：

  - REAL_RETURN_CRITIC_ANCHOR=1
  - REAL_ACTOR_PPO_EVERY>0
  - 或 joint real-grounded critic 相关标志

  当前 Sokoban/FrozenLake runner 均未设置这些标志。最新 PPO 的真实在线 episode 用于 unified replay，而不是直接转换成 real-return PPO batch。

  因此它不是当前训练路径，但仍作为条件兼容实现留在 mbrl_train.py 中。

  ## 五、保留为诊断工具，不参与最新正式参数更新

  以下 16 个文件并非“废代码”；它们用于定位问题、临时拟合 probe 或生成报告，但正常 formal runner 不会执行：

  - rl/action_q_probe.py
  - rl/counterfactual_action_audit.py
  - rl/critic_warmup_probe.py
  - rl/fixed_counterfactual_q_probe.py
  - rl/fixed_scalar_v_probe.py
  - rl/frozen_actor_critic_cross_diagnostic.py
  - rl/frozen_posterior_spatial_probe.py
  - rl/imagined_real_action_ranking_audit.py
  - rl/joint_critic_causal_diagnostic.py
  - rl/near_terminal_ranking_audit.py
  - rl/online_counterfactual_256_audit.py
  - rl/posterior_target_bucket_diagnostic.py
  - rl/prior_reward_decomposition_audit.py
  - rl/spatial_decoder_real_posterior_audit.py
  - rl/terminal_reward_coordinate_audit.py
  - rl/transition_target_causal_diagnostic.py

  共同特征：

  - 大多由 scripts/train_mbrl.py 的命令行参数或环境变量条件触发；
  - 默认不运行；
  - 多数运行后立即退出，不继续 PPO；
  - 有些会训练临时 probe，但不修改正式 Actor/WM；
  - 用于验证空间信息、reward coordinate、imagined-real ranking、Critic 因果关系等。

  这组更合适标为“保留的研究诊断”，而不是“最新训练使用”或“完全弃用”。

  ## 六、旧训练/repair 分支，最新正式流程已经不用

  以下 12 个文件包含实际训练、repair、warmup 或 cache 构建能力，但属于 8 月旧的 latent/runtime-matched/near-terminal 实验链，9 月的 Qwen+V-JEPA formal runner 不再调
  用：

  - rl/counterfactual_spatial_decoder_repair.py
  - rl/formal_q_pretrain.py
  - rl/full_trajectory_h8.py
  - rl/initial_h12_posterior_cache.py
  - rl/long_horizon_critic_grounding.py
  - rl/near_terminal_actor_warmup.py
  - rl/near_terminal_critic_repair.py
  - rl/online_actor_warmup.py
  - rl/runtime_matched_h2_sequence_cache.py
  - rl/runtime_matched_spatial_decoder.py
  - rl/runtime_matched_state_action_prior_repair.py
  - rl/symbolic_sokoban.py

  其中可以进一步区分：

  - counterfactual_spatial_decoder_repair.py、runtime_matched_*：旧 spatial decoder/state-action prior 修复链。
  - near_terminal_*、formal_q_pretrain.py：旧 Critic/Actor 预训练与 repair 链。
  - full_trajectory_h8.py、initial_h12_posterior_cache.py：旧固定 cache 实验。
  - long_horizon_critic_grounding.py：旧真实长轨迹 Critic grounding 分支。
  - online_actor_warmup.py：仍有可选 hook，但最新 runner 未启用。
  - symbolic_sokoban.py：最新 runner 明确排除。

  正式 PPO 脚本明确设置：

  SYMBOLIC_PHYSICS_TERMINAL=0
  CONTINUE_SYMBOLIC_H2_RUN=0

  见 code/MBRL0901/scripts/run_qwen_vjepa_ppo.sh:171。

  所以 symbolic Sokoban 已明确退出最新训练定义；当前 target 必须来自 learned WM，而不是显式 Sokoban 物理规则。

  ## 七、32 个 RL 文件的最终数量归类

   类型                           数量    是否进入最新训练
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━  ━━━━━━━━━━━━━━━━━━
   formal 主训练核心                 2    是
  ─────────────────────────────  ──────  ──────────────────
   框架保留但当前关闭                2    否
  ─────────────────────────────  ──────  ──────────────────
   只读审计/探针                    16    否
  ─────────────────────────────  ──────  ──────────────────
   旧 repair/warmup/cache 路径      12    否
  ─────────────────────────────  ──────  ──────────────────
   合计                             32    —

  真正需要作为当前 Sokoban/FrozenLake formal framework 核心维护的是：

  - rl/counterfactual_h2_ppo.py
  - rl/critic_h2_cache.py

  此外，因为 import/兼容关系，暂时还不能直接删除：

  - rl/offline_bc.py
  - rl/real_return_anchor.py
  - diagnostics/frozenlake_prior_repair.py

  如果目标是清理代码，比较合理的做法是保留 formal 主链、把 16 个诊断模块移动到独立 diagnostics/ 或 research/legacy/，再将 12 个旧训练分支整体归档。不能只删除文件，因为
  train_mbrl.py 中仍保留了对应的条件入口。

─ Worked for 5m 06s ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


› 现在帮我确定最终这两个子文件夹中真正还有用的代码文件有哪些，列出来


• Context compacted

• 我会按“最新 Sokoban/FrozenLake 正式训练链路是否直接调用、是否是仍被调用的依赖”来做最终筛选；仅用于历史实验、一次性诊断或已关闭开关的文件不算“真正还在用”。我再核对一
  次入口脚本与导入关系，然后给你两个子目录的保留清单。

• Explored
  └ Search (diagnostics\.|rl\.|from sembelief_wm\.(diagnostics|rl)|import sembelief_wm\.(diagnostics|rl)|counterfactual_h2_ppo|critic_h2_cache|offline_bc|
           real_return_anchor|frozenlake_lora_spatial_repair|frozenlake_spatial|frozenlake_prior_repair|sokoban_prior_repair_v2|sokoban_stage1_audit) in scripts
    Read run_qwen_vjepa_ppo.sh, run_qwen_vjepa_critic_warmup.sh, run_frozenlake_training.sh, run_frozenlake_prior_spatial_repair.sh, run_qwen_vjepa_prior_repair_v2.py,
         audit_qwen_vjepa_teacher_wm.py

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 这里有一个需要区分的边界：rl/ 里除了前面讨论的 32 个新增实验文件，还有原本就存在、正式训练必需的 PPO、GAE、策略和轨迹实现。我会把“两个目录最终应保留的全部生产代码”列
  出来，而不是只列新增文件；测试文件会单独说明，不混入生产清单。

• Explored
  └ Search *.py in diagnostics
    Search ^from \.|^import  in *.py
    Read mbrl_train.py
    Read mbrl_train.py
    Read assemble.py
    Search offline-bc|behavior-(kl|bc)|counterfactual-action-audit|real-return|critic-h2-cache|frozenlake-spatial-audit in run_qwen_vjepa_ppo.sh
    Search *qwen*vjepa* in scripts
    Search prior_repair_v2|stage1.*audit|audit_qwen|run_qwen_vjepa_prior_repair_v2|run_qwen_vjepa_critic_warmup|run_qwen_vjepa_ppo in scripts
    Search PRE_PPO_REWARD_ACTION_GATE|REAL_RETURN_CRITIC_ANCHOR|REAL_ACTOR_PPO_EVERY|REAL_ACTOR_PPO|REAL_RETURN|COUNTERFACTUAL_ACTION_AUDIT in run_qwen_vjepa_ppo.sh
    Search from \.{1,2}rl\.(action_adapter|gae|llm_policy|policy|ppo|trajectory)|from sembelief_wm\.rl\.(action_adapter|gae|llm_policy|policy|ppo|trajectory) in
           sembelief_wm
    Read __init__.py

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 按 2026-09-05 最新 Sokoban/FrozenLake 正式训练入口，我建议最终保留如下。

  ## 1. sembelief_wm/diagnostics/

  应保留 5 个文件：

  ### Sokoban

  - code/MBRL0901/sembelief_wm/diagnostics/sokoban_prior_repair_v2.py
    Stage 1 的 prior LoRA 修复训练。

  - code/MBRL0901/sembelief_wm/diagnostics/sokoban_stage1_audit.py
    Stage 1 正式验收；同时也是 sokoban_prior_repair_v2.py 的依赖。

  ### FrozenLake

  - code/MBRL0901/sembelief_wm/diagnostics/frozenlake_lora_spatial_repair.py
    当前 Stage 1 spatial prior repair v3 的实际训练实现。

  - code/MBRL0901/sembelief_wm/diagnostics/frozenlake_spatial.py
    当前 Stage 1 held-out spatial/counterfactual 最终验收实现，由 --frozenlake-spatial-audit 调用。

  - code/MBRL0901/sembelief_wm/diagnostics/frozenlake_prior_repair.py
    旧的独立 repair 流程已经不再使用，但当前 frozenlake_lora_spatial_repair.py 仍从中导入 _posterior_rows，所以暂时必须保留。后续可把该函数迁入新文件后删除。

  可以归档或删除的 2 个：

  - frozenlake_h1_h2_repair.py
  - frozenlake_h2_audit.py

  它们属于旧的 H1/H2 FrozenLake 修复/审计路线，已经被 spatial repair v3 和 frozenlake_spatial.py 取代。

  ———

  ## 2. sembelief_wm/rl/

  ### 正式训练核心：应保留 9 个

  基础 RL 框架：

  - code/MBRL0901/sembelief_wm/rl/__init__.py
  - code/MBRL0901/sembelief_wm/rl/action_adapter.py
  - code/MBRL0901/sembelief_wm/rl/trajectory.py
  - code/MBRL0901/sembelief_wm/rl/policy.py
  - code/MBRL0901/sembelief_wm/rl/llm_policy.py
  - code/MBRL0901/sembelief_wm/rl/gae.py
  - code/MBRL0901/sembelief_wm/rl/ppo.py

  最新 exact-H2 训练新增核心：

  - code/MBRL0901/sembelief_wm/rl/counterfactual_h2_ppo.py
    最新 Sokoban/FrozenLake Stage 3/4 的 exact-H2 PPO batch 构造；被训练管线直接导入。

  - code/MBRL0901/sembelief_wm/rl/critic_h2_cache.py
    Stage 3 Critic warmup 的固定、level-disjoint H2 cache 加载与安装。

  ### 暂时需要保留，但功能未启用：1 个

  - code/MBRL0901/sembelief_wm/rl/offline_bc.py

  最新两个正式脚本都设置：

  offline_bc_steps=0
  behavior_kl_coef=0
  behavior_bc_coef=0

  因此 BC 功能没有参与最新训练。但 code/MBRL0901/sembelief_wm/pipelines/assemble.py:27 顶层仍直接导入它，现在直接删文件会导致训练启动失败。

  结论：当前源码不修改时要保留；若清理 assemble.py 中的导入及 BC 分支，则可以删除。

  ### 最新正式训练不需要的 30 个新增文件

  以下文件不再进入当前 Sokoban/FrozenLake 正式训练参数更新链路：

  action_q_probe.py
  counterfactual_action_audit.py
  counterfactual_spatial_decoder_repair.py
  critic_warmup_probe.py
  fixed_counterfactual_q_probe.py
  fixed_scalar_v_probe.py
  formal_q_pretrain.py
  frozen_actor_critic_cross_diagnostic.py
  frozen_posterior_spatial_probe.py
  full_trajectory_h8.py
  imagined_real_action_ranking_audit.py
  initial_h12_posterior_cache.py
  joint_critic_causal_diagnostic.py
  long_horizon_critic_grounding.py
  near_terminal_actor_warmup.py
  near_terminal_critic_repair.py
  near_terminal_ranking_audit.py
  online_actor_warmup.py
  online_counterfactual_256_audit.py
  posterior_target_bucket_diagnostic.py
  prior_reward_decomposition_audit.py
  real_return_anchor.py
  runtime_matched_h2_sequence_cache.py
  runtime_matched_spatial_decoder.py
  runtime_matched_state_action_prior_repair.py
  spatial_decoder_real_posterior_audit.py
  symbolic_sokoban.py
  terminal_reward_coordinate_audit.py
  transition_target_causal_diagnostic.py

  其中：

  - 一部分是一次性 probe/audit，不产生正式训练参数更新。
  - 一部分是旧的 actor/critic repair、cache 或 Q pretrain 路线。
  - symbolic_sokoban.py 在最新 runner 中被明确关闭。
  - real_return_anchor.py 只存在于未启用的条件分支。
  - counterfactual_action_audit.py 只在未启用的 PRE_PPO_REWARD_ACTION_GATE 下调用。

  ## 最终数量

   目录            当前目录文件数    真正应保留                                        兼容性暂留
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   diagnostics/                 7             5    其中 frozenlake_prior_repair.py 是共享函数依赖
  ──────────────  ────────────────  ────────────  ────────────────────────────────────────────────
   rl/                         39             9                            offline_bc.py 额外暂留
  ──────────────  ────────────────  ────────────  ────────────────────────────────────────────────
   合计                        46            14                      当前源码原样运行需保留 15 个

  所以，若目标是整理成最新正式训练的最小代码集：

  - 最终功能代码：14 个。
  - 暂不修改调用方时：保留 15 个。
  - 前面讨论的“新增 7+32”共 39 个文件里，真正进入最新流程的是 7 个：diagnostics 5 个、RL 2 个。offline_bc.py 只是当前导入结构要求的过渡保留项。

─ Worked for 2m 43s ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


› Summarize recent commits

  gpt-5.6-sol high · /personal/jiayu2026
