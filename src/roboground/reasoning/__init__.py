"""推理层：空间关系、规则引擎、VLM 后端。"""

from roboground.reasoning.spatial_relations import (
    RELATION_NAMES,
    bbox_gap,
    center_distance,
    compute_all_relations,
    compute_relation,
    containment_ratio,
    dominant_relation,
    find_relations,
)
from roboground.reasoning.rule_engine import RuleEngine, parse_intent
from roboground.reasoning.vlm import HybridReasoner, VLMBrain, build_vlm_prompt, extract_json

__all__ = [
    # 空间关系
    "RELATION_NAMES",
    "center_distance",
    "bbox_gap",
    "containment_ratio",
    "dominant_relation",
    "compute_relation",
    "compute_all_relations",
    "find_relations",
    # 规则引擎
    "RuleEngine",
    "parse_intent",
    # VLM
    "VLMBrain",
    "HybridReasoner",
    "build_vlm_prompt",
    "extract_json",
]
