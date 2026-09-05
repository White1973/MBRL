"""Fail-closed compatibility lock for the few components Le-WM still shares."""
from __future__ import annotations

import hashlib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]

# Updating an entry is an explicit compatibility event: run both ordinary-WM
# and Le-WM suites, inspect checkpoint semantics, then update the digest in the
# same reviewed change. Ordinary-WM work must never update this automatically.
LOCKED_SHARED_CORE_V1 = {
    "sembelief_wm/config.py": "77cc543511eee0ccd2eaa970c4df41a8f1f5ae68fb53da082170cb4b26f7b39b",
    "sembelief_wm/data/datasource.py": "eb755e727e0467b370d4b0d8110e0b1aa15eff2a71af9d8054e2666c559ab56c",
    "sembelief_wm/data/tokenizers/image.py": "c4396f344dd8ab6cb0bcece44c9676fdeb0259a3489d1bc0281bffca01c52d8f",
    "sembelief_wm/data/adapters/sokoban.py": "dc3a19e121e1f8405152c5ad305c868784c78af988aa36e1f2f35c6ca0483d0f",
    "sembelief_wm/envs/sokoban/action_adapter.py": "dd13621e809def2892e1b25297be3e502da625b56da4f1a91ad261091a912dac",
    "sembelief_wm/model/__init__.py": "1b9f7a94021bf99d3ad7904d3de852300c19c50853c1f47f3dbf804a487d08e7",
    "sembelief_wm/model/backbone_qwen.py": "e3b54dc2ae90f7059756dc464ec0098957f9e434c5af009806b739eae7a4c83a",
    "sembelief_wm/model/belief.py": "ee7449a8ff89d37e1f96421807087d154e9e1b8e96bafa9c3dcc90aba3a48d57",
    "sembelief_wm/model/checkpoint_semantics.py": "6fb418e861cca974b6e0fe65d1999ec13df59ee9642891e57d89eff1a2cad1d4",
    "sembelief_wm/model/policy_backbone.py": "dce11d18753905e6d078c0fa29f4ebb7a6f35abaee36e182108655ee372e6229",
    "sembelief_wm/model/reward.py": "6003c3f124f14beec6d5fb09607f4ee1432540d19c12fe0ef348b0c5c8bd3972",
    "sembelief_wm/model/world_model.py": "659ffd3a094dde5cdf8d3f751fe6cf1f518db0409d12bc6b41326d305e2dbffc",
    "sembelief_wm/model/transition.py": "2e15eb74d8a4b5b86e19ed4fc8f7d31ab0d164c950c61cd66ddf441760f80ba3",
    "sembelief_wm/types.py": "e3ca5c34043bdc43aeff460449ef734ca8fce2ce97720eb2f8ca08c560ac49f4",
    "sembelief_wm/collectors/real.py": "db93eaad878f1de1ab23274c80e3b7edce0d9d632c24a5341c8f86c4ca9b52ed",
    "sembelief_wm/rl/llm_policy.py": "d2961a948724938981bd276440048b7a366596d49561bdd1f97aa6dc5213dcc7",
    "sembelief_wm/rl/ppo.py": "56c1caa48edcce205512f3b8f5eafe18b6bb8b616e00df3e93c13da4dcbebe06",
    "sembelief_wm/pipelines/assemble.py": "a8d77a327fdc54008dbbd10d93e42320ba8cf2d372a9b60da94de1a9a90a87e1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_shared_core() -> dict[str, str]:
    changed: dict[str, str] = {}
    for relative, expected in LOCKED_SHARED_CORE_V1.items():
        path = _ROOT / relative
        actual = "missing" if not path.is_file() else _sha256(path)
        if actual != expected:
            changed[relative] = f"expected={expected}, actual={actual}"
    if changed:
        details = "\n".join(
            f"  {name}: {value}" for name, value in changed.items()
        )
        raise RuntimeError(
            "Le-WM locked shared-core contract changed. Ordinary-WM edits "
            "cannot silently enter a Le-WM run. Run both compatibility suites "
            "and explicitly revise LOCKED_SHARED_CORE_V1 before proceeding:\n"
            + details
        )
    return dict(LOCKED_SHARED_CORE_V1)


if __name__ == "__main__":
    locked = verify_locked_shared_core()
    print(f"Le-WM shared-core isolation contract passed: {len(locked)} files locked")
