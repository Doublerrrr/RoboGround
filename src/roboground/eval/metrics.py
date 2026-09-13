"""评测指标：3D 定位精度、检测指标、查询命中率。

为什么自己实现 3D IoU
--------------------
SUN RGB-D / ScanNet 的框是**带 yaw 的有向盒**。用轴对齐近似会让
"斜放的椅子"的 IoU 被严重低估，导致评测结果不可信。
所以这里实现了**正确的 yaw 有向盒 3D IoU**：
- BEV（俯视 x-y 平面）：用 Sutherland–Hodgman 多边形裁剪求两个旋转矩形的交面积；
- z 方向：区间交集。

这样得到的 IoU 与标准评测（如 SUN RGB-D 官方脚本）在语义上一致。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ==========================================================================
# 有向盒几何
# ==========================================================================
def rotz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotated_rect_corners(
    center_xy: Sequence[float],
    size_xy: Sequence[float],
    yaw: float,
) -> np.ndarray:
    """旋转矩形（BEV）的 4 个角点，按逆时针顺序。"""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    hx, hy = abs(float(size_xy[0])) / 2.0, abs(float(size_xy[1])) / 2.0
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
    return local @ rotz(yaw).T + np.array([cx, cy], dtype=np.float64)


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman：用凸多边形 `clip` 裁剪 `subject`（均为逆时针）。"""

    def inside(p, a, b) -> bool:
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

    def intersect(p, q, a, b) -> np.ndarray:
        x1, y1 = p
        x2, y2 = q
        x3, y3 = a
        x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return q
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return np.array([px, py], dtype=np.float64)

    output = [np.asarray(p, dtype=np.float64) for p in subject]
    clip_list = [np.asarray(p, dtype=np.float64) for p in clip]
    if len(output) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    for i in range(len(clip_list)):
        a = clip_list[i]
        b = clip_list[(i + 1) % len(clip_list)]
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(intersect(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(intersect(s, e, a, b))
            s = e

    return np.asarray(output, dtype=np.float64) if output else np.zeros((0, 2), dtype=np.float64)


def polygon_area(poly: np.ndarray) -> float:
    """鞋带公式求多边形面积（逆时针为正）。"""
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def box3d_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """两个 yaw 有向 3D 盒的 IoU。

    Parameters
    ----------
    box_a, box_b : (7,) 或 (6,)
        `[cx, cy, cz, dx, dy, dz, (yaw)]`。
    """
    a = np.asarray(box_a, dtype=np.float64).reshape(-1)
    b = np.asarray(box_b, dtype=np.float64).reshape(-1)
    if a.size < 6 or b.size < 6:
        return 0.0

    a_yaw = float(a[6]) if a.size > 6 else 0.0
    b_yaw = float(b[6]) if b.size > 6 else 0.0

    # --- z 方向：区间交集 ---
    a_z1, a_z2 = a[2] - abs(a[5]) / 2.0, a[2] + abs(a[5]) / 2.0
    b_z1, b_z2 = b[2] - abs(b[5]) / 2.0, b[2] + abs(b[5]) / 2.0
    z_inter = max(0.0, min(a_z2, b_z2) - max(a_z1, b_z1))
    if z_inter <= 0:
        return 0.0

    # --- BEV：旋转矩形交面积 ---
    poly_a = rotated_rect_corners(a[:2], a[3:5], a_yaw)
    poly_b = rotated_rect_corners(b[:2], b[3:5], b_yaw)
    inter_poly = _clip_polygon(poly_a, poly_b)
    inter_area = polygon_area(inter_poly)

    vol_a = abs(a[3]) * abs(a[4]) * abs(a[5])
    vol_b = abs(b[3]) * abs(b[4]) * abs(b[5])
    inter_vol = inter_area * z_inter
    union = vol_a + vol_b - inter_vol
    return float(inter_vol / union) if union > 1e-12 else 0.0


def box3d_center(box: np.ndarray) -> np.ndarray:
    return np.asarray(box, dtype=np.float64).reshape(-1)[:3]


# ==========================================================================
# 检测指标
# ==========================================================================
def match_boxes(
    pred_boxes: Sequence[np.ndarray],
    gt_boxes: Sequence[np.ndarray],
    iou_threshold: float = 0.25,
) -> Dict[str, Any]:
    """贪心匹配（按 IoU 从高到低），返回 TP/FP/FN 与匹配对。"""
    preds = list(pred_boxes)
    gts = list(gt_boxes)
    if not preds:
        return {"tp": 0, "fp": 0, "fn": len(gts), "matches": [], "ious": []}
    if not gts:
        return {"tp": 0, "fp": len(preds), "fn": 0, "matches": [], "ious": []}

    iou_matrix = np.zeros((len(preds), len(gts)), dtype=np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_matrix[i, j] = box3d_iou(p, g)

    matches: List[Tuple[int, int]] = []
    ious: List[float] = []
    used_gt = set()
    used_pred = set()

    order = np.dstack(np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape))[0]
    for pi, gi in order:
        if iou_matrix[pi, gi] < iou_threshold:
            break
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(int(pi))
        used_gt.add(int(gi))
        matches.append((int(pi), int(gi)))
        ious.append(float(iou_matrix[pi, gi]))

    return {
        "tp": len(matches),
        "fp": len(preds) - len(matches),
        "fn": len(gts) - len(matches),
        "matches": matches,
        "ious": ious,
    }


def detection_metrics(
    all_preds: Sequence[Sequence[np.ndarray]],
    all_gts: Sequence[Sequence[np.ndarray]],
    *,
    iou_thresholds: Sequence[float] = (0.25, 0.50),
) -> Dict[str, float]:
    """跨场景聚合检测指标（precision / recall / F1 / mIoU）。"""
    out: Dict[str, float] = {"num_scenes": float(len(all_preds))}
    for thr in iou_thresholds:
        tp = fp = fn = 0
        ious: List[float] = []
        for preds, gts in zip(all_preds, all_gts):
            m = match_boxes(preds, gts, iou_threshold=float(thr))
            tp += m["tp"]
            fp += m["fp"]
            fn += m["fn"]
            ious.extend(m["ious"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        tag = f"@{thr:.2f}"
        out[f"tp{tag}"] = float(tp)
        out[f"fp{tag}"] = float(fp)
        out[f"fn{tag}"] = float(fn)
        out[f"precision{tag}"] = float(precision)
        out[f"recall{tag}"] = float(recall)
        out[f"f1{tag}"] = float(f1)
        out[f"mean_iou{tag}"] = float(np.mean(ious)) if ious else 0.0
    return out


# ==========================================================================
# 定位误差
# ==========================================================================
def localization_errors(
    pred_centers: Sequence[np.ndarray],
    gt_centers: Sequence[np.ndarray],
) -> Dict[str, float]:
    """中心点定位误差统计（米）。

    这是本项目最能体现"空间语义精度"的指标 —— 机器人不在乎 IoU，
    在乎"你说杯子在这，误差多少厘米"。
    """
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return {"n": 0.0, "median_m": float("nan"), "mean_m": float("nan")}

    n = min(len(pred_centers), len(gt_centers))
    p = np.stack([np.asarray(c, dtype=np.float64).reshape(3) for c in pred_centers[:n]], axis=0)
    g = np.stack([np.asarray(c, dtype=np.float64).reshape(3) for c in gt_centers[:n]], axis=0)
    d = np.linalg.norm(p - g, axis=1)
    return {
        "n": float(n),
        "median_m": float(np.median(d)),
        "mean_m": float(d.mean()),
        "p90_m": float(np.percentile(d, 90)),
        "min_m": float(d.min()),
        "max_m": float(d.max()),
        "within_0.25m": float((d <= 0.25).mean()),
        "within_0.50m": float((d <= 0.50).mean()),
    }


def topk_accuracy(
    pred_labels: Sequence[Sequence[str]],
    gt_labels: Sequence[str],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, float]:
    """Top-K 命中率（用于查询/语言定位评测）。"""
    out: Dict[str, float] = {"num_queries": float(len(gt_labels))}
    for k in ks:
        hits = 0
        for preds, gt in zip(pred_labels, gt_labels):
            topk = [str(x).lower() for x in list(preds)[: int(k)]]
            if str(gt).lower() in topk:
                hits += 1
        out[f"top{k}"] = hits / max(len(gt_labels), 1)
    return out


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """把多条记录按 key 求均值（忽略 nan）。"""
    if not rows:
        return {}
    keys: set = set()
    for r in rows:
        keys.update(r.keys())
    out: Dict[str, float] = {}
    for k in sorted(keys):
        vals = [float(r[k]) for r in rows if k in r and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def format_metrics(metrics: Dict[str, float], *, title: Optional[str] = None) -> str:
    """把指标 dict 渲染成对齐的文本表（demo/报告输出）。"""
    lines: List[str] = []
    if title:
        lines.append(title)
    width = max((len(k) for k in metrics), default=0)
    for k, v in metrics.items():
        if isinstance(v, float) and np.isfinite(v):
            lines.append(f"  {k.ljust(width)} : {v:.4f}")
        else:
            lines.append(f"  {k.ljust(width)} : {v}")
    return "\n".join(lines)
