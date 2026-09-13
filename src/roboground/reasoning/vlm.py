"""VLM 空间推理后端（Qwen2-VL 等），**懒加载 + 优雅降级**。

设计立场（面试要讲的观点）
------------------------
**VLM 不应该替代几何，而应该做几何做不到的事。**
- 几何/规则引擎擅长：精确的米制距离、包含关系判定、可复现、毫秒级；
- VLM 擅长：理解开放式表达（"那个放在电器旁边的小东西"）、
  跨模态常识（"这个能装水吗"）、自然语言组织。

所以本项目的架构是 **规则引擎保底 + VLM 增强**：
1. 规则引擎先算出一份**带精确证据**的结果；
2. 把"地图摘要 + 规则引擎结果"作为上下文喂给 VLM，让它在**有事实约束**
   的前提下组织语言或做开放式推理；
3. 若 VLM 不可用（未装依赖 / 未下权重 / 显存不足），系统无缝退回规则引擎。

为什么必须让 VLM 输出 JSON
-------------------------
自由文本无法被下游规划模块消费，也无法验证是否幻觉。所以这里强制
"结构化输出 + 数值必须来自给定上下文"，并对解析失败做兜底。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from roboground.types import ReasoningResult, SpatialRelation
from roboground.utils.logging import get_logger

logger = get_logger("reasoning.vlm")


# ==========================================================================
# 提示词与解析
# ==========================================================================
SYSTEM_PROMPT = (
    "你是服务机器人的空间推理模块。你会收到一份 3D 语义地图的结构化摘要，"
    "以及用户关于场景的问题。\n"
    "规则：\n"
    "1. 只能使用摘要中出现的物体与坐标，**绝对不要编造**物体或数值；\n"
    "2. 所有距离单位是米，保留两位小数；\n"
    "3. 如果摘要里没有相关信息，就直说找不到；\n"
    "4. 必须输出**严格的 JSON**，不要输出任何额外文字或 markdown 代码块。\n"
    'JSON 格式：{"answer": "中文回答", '
    '"targets": [{"label": "物体名", "position": [x, y, z]}], '
    '"relations": [{"subject": "A", "relation": "above/below/near/...", "object": "B"}], '
    '"confidence": 0.0~1.0}'
)


def build_vlm_prompt(
    query: str,
    map_summary: str,
    *,
    relations_text: str = "",
    rule_hint: Optional[str] = None,
) -> str:
    """构造喂给 VLM 的用户提示词（把结构化上下文拼进去）。"""
    blocks: List[str] = ["【3D 语义地图摘要】", map_summary.strip() or "(空地图)"]

    if relations_text.strip():
        blocks += ["", "【已计算的空间关系（精确值，可直接引用）】", relations_text.strip()]

    if rule_hint:
        blocks += [
            "", "【几何规则引擎的计算结果（精确，可信）】", rule_hint.strip(),
            "请以它为准，不要修改其中的数值。",
        ]

    blocks += ["", "【用户问题】", str(query).strip()]
    return "\n".join(blocks)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出里稳健地抽出 JSON（容忍 markdown 围栏、前后废话、尾逗号）。"""
    if not text:
        return None
    candidates: List[str] = []

    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))

    # 最外层花括号配对
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])

    candidates.append(text)

    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                data = json.loads(attempt)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return None


# ==========================================================================
# VLM 后端
# ==========================================================================
class VLMBrain:
    """Qwen2-VL（或兼容的视觉语言模型）后端。

    8GB 显存下的配置建议（已在 config 里默认给出）：
    - `model_id = "Qwen/Qwen2-VL-2B-Instruct"`（2B 才放得下）
    - `load_in_4bit = True`（bitsandbytes 4bit 量化，显存约 2-3GB）
    - 推理时**只喂文本上下文**，图像可选（喂图会显著增加视觉 token）

    Examples
    --------
    >>> brain = VLMBrain()                    # doctest: +SKIP
    >>> brain.is_available()                  # doctest: +SKIP
    False
    """

    name = "vlm"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        *,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 256,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_new_tokens = int(max_new_tokens)
        self.load_in_4bit = bool(load_in_4bit)
        self.device_map = device_map
        self.temperature = float(temperature)
        self._model = None
        self._processor = None
        self._load_error: Optional[str] = None

    # ---------------- 可用性探测 ----------------
    def is_available(self, *, try_load: bool = False) -> bool:
        """依赖是否就绪。`try_load=True` 时会真的尝试加载（慢）。"""
        try:
            import torch  # noqa: F401, PLC0415
            import transformers  # noqa: F401, PLC0415
        except ImportError as exc:
            self._load_error = f"缺少依赖：{exc}"
            return False

        if try_load:
            try:
                self._ensure_loaded()
            except Exception as exc:
                self._load_error = str(exc)
                return False
        return True

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ---------------- 懒加载 ----------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoProcessor  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "VLMBrain 需要 transformers 与 torch：pip install -e \".[vlm]\""
            ) from exc

        logger.info(f"加载 VLM：{self.model_id}（4bit={self.load_in_4bit}）")

        model_cls = None
        for cls_name in ("Qwen2VLForConditionalGeneration", "AutoModelForVision2Seq",
                         "AutoModelForCausalLM"):
            try:
                import transformers  # noqa: PLC0415

                model_cls = getattr(transformers, cls_name, None)
                if model_cls is not None:
                    break
            except Exception:
                continue
        if model_cls is None:
            raise RuntimeError("当前 transformers 版本里找不到可用的视觉语言模型类；请升级 transformers")

        kwargs: Dict[str, Any] = {"device_map": self.device_map}
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig  # noqa: PLC0415

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as exc:
                logger.warn(f"4bit 量化不可用（{exc}），改为 fp16 加载")
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float16

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = model_cls.from_pretrained(self.model_id, **kwargs)

        if self.adapter_path:
            try:
                from peft import PeftModel  # noqa: PLC0415

                self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
                logger.info(f"已加载 LoRA 适配器：{self.adapter_path}")
            except Exception as exc:
                logger.warn(f"加载 LoRA 失败（{exc}），使用基座模型")

        self._model.eval()

    def warmup(self) -> None:
        self._ensure_loaded()

    # ---------------- 推理 ----------------
    def generate(
        self,
        prompt: str,
        *,
        image: Optional[np.ndarray] = None,
        system: str = SYSTEM_PROMPT,
    ) -> str:
        """生成文本（可带一张图）。"""
        self._ensure_loaded()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        messages: List[Dict[str, Any]] = [{"role": "system", "content": [{"type": "text", "text": system}]}]
        user_content: List[Dict[str, Any]] = []
        if image is not None:
            user_content.append({"type": "image", "image": Image.fromarray(np.asarray(image, dtype=np.uint8))})
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if image is not None:
            inputs = self._processor(
                text=[text],
                images=[Image.fromarray(np.asarray(image, dtype=np.uint8))],
                return_tensors="pt", padding=True,
            )
        else:
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)

        device = next(self._model.parameters()).device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": self.temperature})
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)

        generated = out[:, inputs["input_ids"].shape[1]:]
        decoded = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0] if decoded else ""

    # ---------------- 高层：回答空间问题 ----------------
    def answer(
        self,
        query: str,
        semantic_map,
        *,
        image: Optional[np.ndarray] = None,
        rule_result: Optional[ReasoningResult] = None,
        relations_text: str = "",
        top_relations: int = 10,
    ) -> ReasoningResult:
        """用 VLM 回答空间问题（规则的精确值作为上下文注入）。"""
        if not self.is_available():
            raise RuntimeError(
                f"VLM 不可用：{self.load_error or '依赖缺失'}。"
                '请安装 pip install -e ".[vlm]" 并确保权重可下载'
            )

        if not relations_text and rule_result is None:
            try:
                from roboground.reasoning.spatial_relations import (  # noqa: PLC0415
                    compute_all_relations,
                    describe_relations,
                )

                relations_text = describe_relations(
                    compute_all_relations(semantic_map.objects, max_distance=3.0),
                    top=top_relations,
                )
            except Exception:
                relations_text = ""

        prompt = build_vlm_prompt(
            query,
            semantic_map.describe(max_objects=25),
            relations_text=relations_text,
            rule_hint=(rule_result.answer if rule_result is not None else None),
        )

        raw = self.generate(prompt, image=image)
        data = extract_json(raw)

        if data is None:
            logger.warn(f"VLM 未输出可解析的 JSON，原始输出前 200 字：{raw[:200]!r}")
            return ReasoningResult(
                query=query,
                answer=(rule_result.answer if rule_result else raw.strip()[:300]),
                targets=(rule_result.targets if rule_result else []),
                relations=(rule_result.relations if rule_result else []),
                distances=(rule_result.distances if rule_result else {}),
                confidence=0.3,
                backend=self.name,
                debug={"raw": raw[:500], "json_parsed": False},
            )

        targets = self._resolve_targets(data.get("targets"), semantic_map)
        relations = self._resolve_relations(data.get("relations"))

        return ReasoningResult(
            query=query,
            answer=str(data.get("answer", "")).strip(),
            targets=targets,
            relations=relations,
            distances=(rule_result.distances if rule_result else {}),
            confidence=float(data.get("confidence", 0.5) or 0.5),
            backend=self.name,
            debug={"raw": raw[:500], "json_parsed": True},
        )

    # ---------------- 结构化输出解析 ----------------
    @staticmethod
    def _resolve_targets(raw_targets: Any, semantic_map) -> List[Any]:
        """把 VLM 输出的 target 名称解析成地图里的真实物体。

        这一步是**幻觉闸门**：VLM 编出来的物体名在这里会被丢掉，
        因为它无法匹配到地图中的任何真实物体。
        """
        out: List[Any] = []
        if not isinstance(raw_targets, list):
            return out
        for item in raw_targets:
            label = None
            if isinstance(item, dict):
                label = item.get("label") or item.get("name")
            elif isinstance(item, str):
                label = item
            if not label:
                continue
            hits = semantic_map.query_text(str(label), top_k=1, level="object")
            if hits and hits[0].obj is not None:
                out.append(hits[0].obj)
        return out

    @staticmethod
    def _resolve_relations(raw_relations: Any) -> List[SpatialRelation]:
        out: List[SpatialRelation] = []
        if not isinstance(raw_relations, list):
            return out
        for item in raw_relations:
            if not isinstance(item, dict):
                continue
            try:
                out.append(SpatialRelation(
                    subject=str(item.get("subject", "")),
                    relation=str(item.get("relation", "unknown")),
                    object=str(item.get("object", "")),
                    distance=float(item.get("distance", 0.0) or 0.0),
                    evidence={"source": 1.0},      # 标记来源为 VLM（无几何证据）
                ))
            except Exception:
                continue
        return out

    def __repr__(self) -> str:
        return f"VLMBrain(model={self.model_id!r}, loaded={self._model is not None})"


# ==========================================================================
# 混合推理：规则保底 + VLM 增强
# ==========================================================================
class HybridReasoner:
    """先跑规则引擎（快、精确），可选让 VLM 改写/扩展答案。"""

    name = "hybrid"

    def __init__(
        self,
        semantic_map,
        *,
        cfg: Any = None,
        vlm: Optional[VLMBrain] = None,
        use_vlm: bool = True,
    ) -> None:
        from roboground.reasoning.rule_engine import RuleEngine  # noqa: PLC0415

        self.map = semantic_map
        self.cfg = cfg
        self.rules = RuleEngine(semantic_map, cfg=cfg)
        self.vlm = vlm
        if self.vlm is None and cfg is not None and str(cfg.get("reasoning.backend", "rules")) == "qwen2vl":
            self.vlm = VLMBrain(**{k: v for k, v in (cfg.get("reasoning.vlm", {}) or {}).items()})
        self.use_vlm = bool(use_vlm) and self.vlm is not None

    def answer(self, query: str, *, image: Optional[np.ndarray] = None) -> ReasoningResult:
        rule_result = self.rules.answer(query)

        if not self.use_vlm or not self.vlm.is_available():
            if self.use_vlm and self.vlm is not None:
                logger.debug(f"VLM 不可用，使用规则引擎结果（{self.vlm.load_error}）")
            return rule_result

        try:
            vlm_result = self.vlm.answer(query, self.map, image=image, rule_result=rule_result)
            # 保留规则的精确数值，用 VLM 的自然语言表达
            vlm_result.distances = rule_result.distances
            if not vlm_result.relations:
                vlm_result.relations = rule_result.relations
            if not vlm_result.targets:
                vlm_result.targets = rule_result.targets
            vlm_result.debug["rule_answer"] = rule_result.answer
            return vlm_result
        except Exception as exc:
            logger.warn(f"VLM 推理失败（{exc}），退回规则引擎")
            rule_result.debug["vlm_error"] = str(exc)
            return rule_result

    def __repr__(self) -> str:
        return f"HybridReasoner(vlm={'on' if self.use_vlm else 'off'}, {self.rules!r})"
