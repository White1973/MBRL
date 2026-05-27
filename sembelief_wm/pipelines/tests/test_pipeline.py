"""Tests for the MBRL training pipeline orchestration."""
import torch
import torch.nn as nn
from torch import Tensor

from sembelief_wm.rl.trajectory import Trajectory
from sembelief_wm.rl.ppo import PPOUpdater, PPOConfig
from sembelief_wm.pipelines.mbrl_train import MBRLPipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakePolicy(nn.Module):
    """Minimal policy for pipeline testing."""

    def __init__(self, state_dim: int = 64, num_actions: int = 4):
        super().__init__()
        self.net = nn.Linear(state_dim, num_actions)
        self.value_head = nn.Linear(state_dim, 1)
        self._state_dim = state_dim

    def act(self, states: Tensor, **kwargs):
        logits = self.net(states)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        value = self.value_head(states).squeeze(-1)
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, states: Tensor, actions: Tensor, **kwargs):
        logits = self.net(states)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_head(states).squeeze(-1)
        return dist.log_prob(actions), dist.entropy(), value

    def parameters(self, recurse=True):
        return super().parameters(recurse=recurse)


def make_fake_imagine_fn(B: int, H: int, state_dim: int, num_actions: int):
    """Returns a fake imagine_fn that produces valid Trajectory."""

    def imagine_fn(start_beliefs: Tensor) -> Trajectory:
        actual_B = start_beliefs.shape[0]
        return Trajectory(
            states=torch.randn(actual_B, H, state_dim),
            actions=torch.randint(0, num_actions, (actual_B, H)),
            rewards=torch.randn(actual_B, H) * 0.1,
            dones=torch.zeros(actual_B, H),
            log_probs=torch.randn(actual_B, H) * -1.0,
            values=torch.randn(actual_B, H + 1) * 0.5,
            mask=None,
        )

    return imagine_fn


def make_fake_sample_beliefs_fn(state_dim: int):
    def sample_beliefs_fn(batch_size: int) -> Tensor:
        return torch.randn(batch_size, state_dim)

    return sample_beliefs_fn


class FakeLogger:
    def __init__(self):
        self.logs: list[tuple[int, dict[str, float]]] = []

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        self.logs.append((step, metrics))


class FakeEvaluator:
    def __init__(self, success_rate: float = 0.15):
        self.success_rate = success_rate
        self.call_count = 0

    def evaluate(self, num_episodes: int) -> dict[str, float]:
        self.call_count += 1
        return {
            "eval/success_rate": self.success_rate,
            "eval/avg_return": self.success_rate * 10.0,
            "eval/num_episodes": float(num_episodes),
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_runs_full_loop():
    """Pipeline runs N updates, logs metrics, evals at correct frequency."""
    state_dim = 64
    num_actions = 4
    policy = FakePolicy(state_dim, num_actions)
    logger = FakeLogger()
    evaluator = FakeEvaluator()

    cfg = PipelineConfig(
        total_updates=6,
        eval_every=3,
        eval_episodes=20,
        checkpoint_every=0,
        rollout_batch_size=8,
        rollout_horizon=4,
        ppo=PPOConfig(
            lr=1e-3,
            epochs=2,
            minibatch_size=4,
        ),
        gamma=0.99,
        gae_lambda=0.95,
    )

    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=make_fake_imagine_fn(8, 4, state_dim, num_actions),
        sample_beliefs_fn=make_fake_sample_beliefs_fn(state_dim),
        ppo_updater=PPOUpdater(policy=policy, config=cfg.ppo),
        policy=policy,
        evaluator=evaluator,
        logger=logger,
    )

    pipeline.train()

    # Should have logged 6 updates
    assert len(logger.logs) == 6
    # Eval should have been called at update 3 and 6
    assert evaluator.call_count == 2

    # Check metrics keys exist
    _, last_metrics = logger.logs[-1]
    assert "ppo/policy_loss" in last_metrics
    assert "ppo/value_loss" in last_metrics
    assert "ppo/entropy" in last_metrics
    assert "rollout/reward_mean" in last_metrics
    assert "timing/total_sec" in last_metrics
    # Last update is eval step, should have eval metrics
    assert "eval/success_rate" in last_metrics


def test_pipeline_no_eval():
    """Pipeline works without evaluator."""
    state_dim = 32
    policy = FakePolicy(state_dim, 4)

    cfg = PipelineConfig(
        total_updates=3,
        eval_every=0,  # disabled
        rollout_batch_size=4,
        rollout_horizon=4,
    )

    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=make_fake_imagine_fn(4, 4, state_dim, 4),
        sample_beliefs_fn=make_fake_sample_beliefs_fn(state_dim),
        ppo_updater=PPOUpdater(policy=policy, config=cfg.ppo),
        policy=policy,
    )

    pipeline.train()  # Should not crash


def test_pipeline_checkpoint(tmp_path):
    """Pipeline saves and loads checkpoints correctly."""
    state_dim = 32
    policy = FakePolicy(state_dim, 4)

    cfg = PipelineConfig(
        total_updates=4,
        eval_every=0,
        checkpoint_every=2,
        rollout_batch_size=4,
        rollout_horizon=4,
    )

    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=make_fake_imagine_fn(4, 4, state_dim, 4),
        sample_beliefs_fn=make_fake_sample_beliefs_fn(state_dim),
        ppo_updater=PPOUpdater(policy=policy, config=cfg.ppo),
        policy=policy,
    )

    pipeline.train(checkpoint_dir=tmp_path)

    # Checkpoint should exist
    ckpt_path = tmp_path / "latest.pt"
    assert ckpt_path.exists()

    # Load checkpoint
    policy2 = FakePolicy(state_dim, 4)
    pipeline2 = MBRLPipeline(
        config=cfg,
        imagine_fn=make_fake_imagine_fn(4, 4, state_dim, 4),
        sample_beliefs_fn=make_fake_sample_beliefs_fn(state_dim),
        ppo_updater=PPOUpdater(policy=policy2, config=cfg.ppo),
        policy=policy2,
    )
    resume_update = pipeline2.load_checkpoint(ckpt_path)
    assert resume_update == 4  # last checkpoint at update 4


def test_pipeline_with_wm_refresh():
    """Pipeline calls WM refresher at correct frequency."""
    state_dim = 32
    policy = FakePolicy(state_dim, 4)

    class FakeRefreshMetrics:
        def __init__(self):
            self.avg_dynamics_loss = 0.05
            self.avg_reward_loss = 0.03
            self.avg_total_loss = 0.1
            self.num_steps = 5

    class FakeRefresher:
        def __init__(self):
            self.refresh_count = 0

        def refresh(self, sample_fn):
            self.refresh_count += 1
            # Verify sample_fn is callable
            assert callable(sample_fn)
            return FakeRefreshMetrics()

        def state_dict(self):
            return {"count": self.refresh_count}

        def load_state_dict(self, d):
            self.refresh_count = d["count"]

    refresher = FakeRefresher()

    def fake_wm_refresh_sample_fn(batch_size: int):
        return None  # FakeRefresher doesn't use it

    cfg = PipelineConfig(
        total_updates=6,
        eval_every=0,
        checkpoint_every=0,
        rollout_batch_size=4,
        rollout_horizon=4,
        wm_refresh_every=2,
        wm_refresh_steps=3,
    )

    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=make_fake_imagine_fn(4, 4, state_dim, 4),
        sample_beliefs_fn=make_fake_sample_beliefs_fn(state_dim),
        ppo_updater=PPOUpdater(policy=policy, config=cfg.ppo),
        policy=policy,
        wm_refresher=refresher,
        wm_refresh_sample_fn=fake_wm_refresh_sample_fn,
    )

    pipeline.train()

    # WM refresh at updates 2, 4, 6 → 3 times
    assert refresher.refresh_count == 3


def test_pipeline_checkpoint_includes_world_model(tmp_path):
    """Checkpoint saves and restores world model weights (critical for alternating)."""
    state_dim = 32

    # Fake world model
    class FakeWorldModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dynamics = nn.Linear(state_dim, state_dim)

    policy = FakePolicy(state_dim, 4)
    wm = FakeWorldModel()

    cfg = PipelineConfig(
        total_updates=2,
        eval_every=0,
        checkpoint_every=1,
        rollout_batch_size=4,
        rollout_horizon=4,
    )

    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=make_fake_imagine_fn(4, 4, state_dim, 4),
        sample_beliefs_fn=make_fake_sample_beliefs_fn(state_dim),
        ppo_updater=PPOUpdater(policy=policy, config=cfg.ppo),
        policy=policy,
        world_model=wm,
    )

    pipeline.train(checkpoint_dir=tmp_path)
    ckpt_path = tmp_path / "latest.pt"
    assert ckpt_path.exists()

    # Verify WM weights are in checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "world_model" in ckpt
    assert "dynamics.weight" in ckpt["world_model"]

    # Load into a fresh WM and verify weights match
    wm2 = FakeWorldModel()
    policy2 = FakePolicy(state_dim, 4)
    pipeline2 = MBRLPipeline(
        config=cfg,
        imagine_fn=make_fake_imagine_fn(4, 4, state_dim, 4),
        sample_beliefs_fn=make_fake_sample_beliefs_fn(state_dim),
        ppo_updater=PPOUpdater(policy=policy2, config=cfg.ppo),
        policy=policy2,
        world_model=wm2,
    )
    pipeline2.load_checkpoint(ckpt_path)

    # Weights should match after load
    for p1, p2 in zip(wm.parameters(), wm2.parameters()):
        assert torch.equal(p1, p2)


def test_pipeline_wm_refresh_receives_sample_fn():
    """WM refresher receives a callable sample_fn, not an int."""
    state_dim = 32
    policy = FakePolicy(state_dim, 4)

    received_args = []

    class InspectRefresher:
        def refresh(self, sample_fn):
            received_args.append(sample_fn)
            # Return a RefreshMetrics-like object
            class M:
                avg_dynamics_loss = 0.0
                avg_reward_loss = 0.0
                avg_total_loss = 0.0
                num_steps = 1
            return M()

        def state_dict(self):
            return {}

        def load_state_dict(self, d):
            pass

    refresher = InspectRefresher()

    def sample_fn(batch_size: int):
        return None

    cfg = PipelineConfig(
        total_updates=2,
        eval_every=0,
        rollout_batch_size=4,
        rollout_horizon=4,
        wm_refresh_every=1,
    )

    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=make_fake_imagine_fn(4, 4, state_dim, 4),
        sample_beliefs_fn=make_fake_sample_beliefs_fn(state_dim),
        ppo_updater=PPOUpdater(policy=policy, config=cfg.ppo),
        policy=policy,
        wm_refresher=refresher,
        wm_refresh_sample_fn=sample_fn,
    )

    pipeline.train()

    # Should have been called twice (updates 1 and 2)
    assert len(received_args) == 2
    # Each call should receive the sample_fn callable, not an int
    for arg in received_args:
        assert callable(arg)


def test_posterior_grounder_protocol_integration():
    """Verify make_posterior_grounder output matches PosteriorGrounder protocol.

    This tests the seam between assemble.py's _Grounder and
    belief_sampler.sample_start_beliefs — the interface mismatch that
    was found in review.
    """
    from sembelief_wm.collectors.belief_sampler import sample_start_beliefs
    from sembelief_wm.types import BeliefState, SequenceBatch

    K, D = 4, 32
    B, T = 3, 5

    # A grounder that matches the PosteriorGrounder protocol:
    # get_initial_belief(batch_size, *, device, dtype) -> obj with .slots
    # posterior_step(*, prev_belief, prev_actions, observation_tokens, env_ids) -> obj with .slots
    class MockGrounder:
        def get_initial_belief(
            self, batch_size: int, *, device=None, dtype=None
        ):
            return BeliefState(
                slots=torch.zeros(batch_size, K, D, device=device, dtype=dtype)
            )

        def posterior_step(
            self,
            *,
            prev_belief,
            prev_actions,
            observation_tokens,
            env_ids,
        ):
            # Just return prev_belief shifted slightly
            return BeliefState(slots=prev_belief.slots + 0.01)

    grounder = MockGrounder()

    # Create a fake SequenceBatch
    batch = SequenceBatch(
        obs_tokens=torch.randn(B, T, K, D),
        actions=torch.randint(0, 4, (B, T)),
        rewards=torch.randn(B, T),
        episode_lengths=torch.tensor([T, T - 1, T]),
        env_ids=None,
    )

    beliefs = sample_start_beliefs(
        batch, grounder, device=torch.device("cpu"), dtype=torch.float32
    )

    assert beliefs.shape == (B, K, D)
    # Beliefs should not be all zeros (posterior_step adds 0.01 per step)
    assert beliefs.abs().sum() > 0


def test_reward_transform_decoupled():
    """Verify reward transforms work without importing agent/phase2."""
    from sembelief_wm.collectors.reward_transforms import make_reward_transform

    logits = torch.tensor([0.0, 2.0, -2.0])

    # sigmoid_affine
    fn = make_reward_transform("sigmoid_affine", positive_value=1.0, negative_value=-1.0)
    result = fn(logits)
    assert result.shape == (3,)
    # sigmoid(0) = 0.5, so -1 + 0.5 * 2 = 0.0
    assert abs(result[0].item()) < 1e-5

    # raw_sigmoid
    fn = make_reward_transform("raw_sigmoid")
    result = fn(logits)
    assert abs(result[0].item()) < 1e-5  # sigmoid(0) - 0.5 = 0

    # clipped_logit
    fn = make_reward_transform("clipped_logit")
    result = fn(torch.tensor([10.0, -10.0]))
    assert result[0].item() == 5.0
    assert result[1].item() == -5.0


def test_llm_policy_pipeline_integration():
    """LLMActorCritic + SokobanActionAdapter works end-to-end through pipeline."""
    from sembelief_wm.rl.llm_policy import LLMActorCritic, LLMPolicyConfig
    from sembelief_wm.envs.sokoban.action_adapter import SokobanActionAdapter

    D = 64  # hidden dim
    K = 4   # num slots

    # Mock backbone: linear transform (B, K, D) -> (B, K, D)
    backbone = nn.Sequential(nn.Linear(D, D))

    action_adapter = SokobanActionAdapter(hidden_dim=D)
    llm_cfg = LLMPolicyConfig(hidden_dim=D, num_slots=K)
    llm_policy = LLMActorCritic(
        backbone=backbone,
        action_adapter=action_adapter,
        config=llm_cfg,
    )

    # Build pipeline with LLM policy directly (it satisfies Policy protocol)
    cfg = PipelineConfig(
        total_updates=3,
        eval_every=0,
        checkpoint_every=0,
        rollout_batch_size=4,
        rollout_horizon=4,
    )

    state_dim = K * D  # flat belief dim

    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=make_fake_imagine_fn(4, 4, state_dim, 4),
        sample_beliefs_fn=make_fake_sample_beliefs_fn(state_dim),
        ppo_updater=PPOUpdater(policy=llm_policy, config=cfg.ppo),
        policy=llm_policy,
    )

    # Should run without errors — LLMActorCritic satisfies Policy protocol
    pipeline.train()


def test_llm_policy_with_shared_backbone_pipeline():
    """LLMActorCritic with shared backbone — extra_params wired to PPOUpdater."""
    from sembelief_wm.rl.llm_policy import LLMActorCritic, LLMPolicyConfig
    from sembelief_wm.envs.sokoban.action_adapter import SokobanActionAdapter

    D = 32
    K = 4

    # "Shared" backbone — simulates QwenPolicyBackbone.from_shared
    shared_linear = nn.Linear(D, D)

    class FakeSharedBackbone(nn.Module):
        """Simulates shared mode: backbone not registered as submodule."""
        def __init__(self, real_backbone):
            super().__init__()
            object.__setattr__(self, "_real", real_backbone)

        def forward(self, x):
            return self._real(x)

        def trainable_parameters(self):
            return [p for p in self._real.parameters() if p.requires_grad]

    backbone = FakeSharedBackbone(shared_linear)
    action_adapter = SokobanActionAdapter(hidden_dim=D)
    llm_cfg = LLMPolicyConfig(hidden_dim=D, num_slots=K)
    llm_policy = LLMActorCritic(
        backbone=backbone,
        action_adapter=action_adapter,
        config=llm_cfg,
    )

    # Capture initial backbone weight
    w_before = shared_linear.weight.data.clone()

    cfg = PipelineConfig(
        total_updates=3,
        eval_every=0,
        checkpoint_every=0,
        rollout_batch_size=4,
        rollout_horizon=4,
    )

    state_dim = K * D

    ppo_updater = PPOUpdater(
        config=cfg.ppo,
        policy=llm_policy,
        extra_params=backbone.trainable_parameters(),
    )

    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=make_fake_imagine_fn(4, 4, state_dim, 4),
        sample_beliefs_fn=make_fake_sample_beliefs_fn(state_dim),
        ppo_updater=ppo_updater,
        policy=llm_policy,
    )

    pipeline.train()

    # Shared backbone weights should have been updated
    w_after = shared_linear.weight.data
    assert not torch.equal(w_before, w_after), (
        "Shared backbone weights were not updated — extra_params not wired correctly"
    )


def test_pipeline_with_real_collector():
    """Pipeline calls real_collector.collect_and_store at correct frequency."""
    from sembelief_wm.collectors.real import (
        RealCollector, RealCollectorConfig, CollectionResult, EpisodeResult,
    )
    from dataclasses import dataclass

    state_dim = 64
    policy = FakePolicy(state_dim, 4)

    @dataclass
    class FakeBelief:
        slots: torch.Tensor

    class FakeEnv:
        def reset(self, *, seed=None):
            return torch.randn(3)
        def step(self, action):
            return torch.randn(3), -0.1, True, {"success": False}

    class FakeTokenizer:
        def tokenize(self, obs):
            return torch.randn(4, 16)

    class FakeWM:
        def get_initial_belief(self, batch_size, *, device, dtype):
            return FakeBelief(slots=torch.randn(batch_size, 4, 16))
        def posterior_step(self, *, prev_belief, prev_actions,
                          observation_tokens, env_ids=None):
            B = observation_tokens.shape[0]
            return FakeBelief(slots=torch.randn(B, 4, 16))

    # Policy that handles (B, K, D) belief slots (flattens internally)
    class SlotPolicy:
        def act(self, states, **kwargs):
            flat = states.reshape(states.shape[0], -1) if states.dim() > 2 else states
            B = flat.shape[0]
            action = torch.randint(0, 4, (B,))
            return action, torch.zeros(B), torch.zeros(B), {}
        def evaluate_actions(self, states, actions, **kwargs):
            flat = states.reshape(states.shape[0], -1) if states.dim() > 2 else states
            B = flat.shape[0]
            return torch.zeros(B), torch.zeros(B), torch.zeros(B)
        def parameters(self, recurse=True):
            return iter([])

    class FakeReplay:
        def __init__(self):
            self.episodes = []
            self.size = 0
        def add_episode(self, eid, ep):
            self.episodes.append(eid)
            self.size += 1

    replay = FakeReplay()
    slot_policy = SlotPolicy()
    real_collector = RealCollector(
        env_factory=lambda seed: FakeEnv(),
        tokenizer=FakeTokenizer(),
        world_model=FakeWM(),
        policy=slot_policy,
        config=RealCollectorConfig(max_steps=1),
        replay_buffer=replay,
        device="cpu",
        dtype=torch.float32,
    )

    cfg = PipelineConfig(
        total_updates=4,
        eval_every=0,
        checkpoint_every=0,
        rollout_batch_size=4,
        rollout_horizon=4,
        collect_every=2,
        collect_episodes=3,
    )

    logger = FakeLogger()
    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=make_fake_imagine_fn(4, 4, state_dim, 4),
        sample_beliefs_fn=make_fake_sample_beliefs_fn(state_dim),
        ppo_updater=PPOUpdater(policy=policy, config=cfg.ppo),
        policy=policy,
        real_collector=real_collector,
        logger=logger,
    )

    pipeline.train()

    # Collection at updates 2 and 4 → 2 × 3 episodes = 6 episodes in replay
    assert replay.size == 6

    # Check collect metrics are logged
    collect_logged = [
        step for step, m in logger.logs if "collect/success_rate" in m
    ]
    assert len(collect_logged) == 2


if __name__ == "__main__":
    test_pipeline_runs_full_loop()
    test_pipeline_no_eval()
    test_pipeline_checkpoint("/tmp/test_pipeline_ckpt")
    test_pipeline_with_wm_refresh()
    test_pipeline_checkpoint_includes_world_model("/tmp/test_pipeline_wm_ckpt")
    test_pipeline_wm_refresh_receives_sample_fn()
    test_posterior_grounder_protocol_integration()
    test_reward_transform_decoupled()
    test_llm_policy_pipeline_integration()
    test_llm_policy_with_shared_backbone_pipeline()
    test_pipeline_with_real_collector()
    print("All pipeline tests passed!")
