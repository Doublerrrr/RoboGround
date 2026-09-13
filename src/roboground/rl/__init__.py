"""强化学习后训练：用**几何规则引擎作为可验证奖励**（RLVR）微调 VLM。

链条::

    3D 语义地图 ──规则引擎──> 精确答案（Oracle）
                                  │
    VLM 采样 G 个回答 ─────────────┤
                                  ↓
                          verifiable reward
                                  ↓
                   组内优势（GRPO，无需 value network）
                                  ↓
                         带 Clip 的策略梯度

与项目其余部分的关系：
- `reward.py` 的 Oracle 来自 `reasoning/rule_engine.py` 与 `spatial_relations.py`
  （和 `scripts/12_gen_spatial_qa.py` 生成 SFT 数据用的是同一套几何逻辑）；
- `grpo.py` 是纯函数形式的目标函数，可离线单测；
- 本模块只负责"把策略模型接到目标函数上"。
"""

from roboground.rl.grpo import (
    GrpoConfig,
    GrpoLoss,
    TrainStats,
    clipped_surrogate,
    group_advantages,
    grpo_loss,
    kl_penalty,
    sequence_ratios,
    token_ratios,
)
from roboground.rl.reward import (
    RewardBreakdown,
    RewardConfig,
    SpatialOracle,
    correctness,
    parse_answer,
    reward_stats,
    spatial_reward,
)
from roboground.rl.trainer import (
    FakePolicy,
    GrpoTrainer,
    PolicyBase,
    RlSample,
    TrainerConfig,
    build_rl_samples,
)

__all__ = [
    # GRPO 目标函数
    "GrpoConfig", "GrpoLoss", "TrainStats", "grpo_loss", "group_advantages",
    "clipped_surrogate", "kl_penalty", "token_ratios", "sequence_ratios",
    # 可验证奖励
    "SpatialOracle", "RewardConfig", "RewardBreakdown", "spatial_reward",
    "parse_answer", "correctness", "reward_stats",
    # 训练循环
    "PolicyBase", "FakePolicy", "RlSample", "TrainerConfig", "GrpoTrainer",
    "build_rl_samples",
]
