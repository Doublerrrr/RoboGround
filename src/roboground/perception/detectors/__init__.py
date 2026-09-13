"""检测器后端。"""

from roboground.perception.detectors.stub import StubDetector
from roboground.perception.detectors.grounding_dino import GroundingDINODetector
from roboground.perception.detectors.yolo import YOLODetector

__all__ = ["StubDetector", "GroundingDINODetector", "YOLODetector"]
