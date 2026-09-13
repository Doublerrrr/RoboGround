"""分割器后端。"""

from roboground.perception.segmenters.box import BoxSegmenter
from roboground.perception.segmenters.sam import SAMSegmenter

__all__ = ["BoxSegmenter", "SAMSegmenter"]
