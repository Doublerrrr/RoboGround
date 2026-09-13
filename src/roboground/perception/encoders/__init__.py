"""特征编码器后端。

| 后端 | 维度 | 支持文本 | 校准 | 说明 |
|---|---|---|---|---|
| `color_hist` | 72 | ❌ | — | 离线可用，颜色+空间描述子，**无模型依赖** |
| `dinov2` | 384/768/1024 | ❌ | — | 密集语义特征，适合分割/匹配类下游 |
| `clip` | 512/768 | ✅ | 需空文本校准 | 图文对齐，但区域级区分度有限（见实测） |
| `siglip` | 768 | ✅ | **模型原生** | sigmoid 损失 → 自带校准概率，拒识更好 |

实测结论（`scripts/11_ablate_clip_pooling.py`，真实 SUN RGB-D）：
CLIP ViT-B/32 的区域特征在"家具类别"上分离度为**负值**（self_sim < other_sim），
Top-1 仅 0.34。所以本项目把 `siglip` 作为**推荐的开放词汇后端**。
"""

from roboground.perception.encoders.color_hist import ColorHistogramEncoder
from roboground.perception.encoders.dinov2 import DINOv2Encoder
from roboground.perception.encoders.clip_encoder import CLIPEncoder
from roboground.perception.encoders.siglip_encoder import SigLIPEncoder

__all__ = ["ColorHistogramEncoder", "DINOv2Encoder", "CLIPEncoder", "SigLIPEncoder"]

