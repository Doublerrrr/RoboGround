"""ROS2 集成（**防御式导入**：没有 rclpy 也能 import 本包与跑测试）。

目录结构
--------
```
ros2/
├── bridge.py   纯 Python 的消息转换层（不依赖 ROS，可单元测试）★
├── nodes.py    三个 ROS2 节点（perception / map / vlm），运行时才 import rclpy
└── README.md   启动方式与话题约定
```

为什么把转换层单独拆出来
----------------------
1. **可测试**：ROS 环境往往装不上（尤其 Windows），但"数据怎么进出 ROS"
   这件事本身是纯数据变换，必须能被单元测试覆盖；
2. **可替换**：以后要换 ROS1、或者换成 DDS/自研中间件，只改 bridge 即可。

话题与消息约定
-------------
| 话题 | 方向 | 消息类型 | 内容 |
|---|---|---|---|
| `/camera/color/image_raw` | 订阅 | `sensor_msgs/Image` | RGB |
| `/camera/depth/image_raw` | 订阅 | `sensor_msgs/Image` | 16 位深度（毫米） |
| `/camera/color/camera_info` | 订阅 | `sensor_msgs/CameraInfo` | 内参 K |
| `/roboground/semantic_map` | 发布 | `std_msgs/String`(JSON) | 语义地图摘要 + 物体列表 |
| `/roboground/query` | 订阅 | `std_msgs/String` | 自然语言问题 |
| `/roboground/answer` | 发布 | `std_msgs/String`(JSON) | 结构化回答 |

> 用 JSON 字符串而不是自定义 msg：避免编译 .msg 依赖，方便跨版本复用。
> 真要在生产里用，建议定义 `roboground_msgs/SemanticMap.msg` 以获得类型安全。
"""

from roboground.deployment.ros2.bridge import (
    ROS2_AVAILABLE,
    TOPIC_TYPES,
    answer_to_dict,
    camera_info_to_intrinsics,
    dict_to_frame,
    frame_to_dict,
    map_to_dict,
    require_ros2,
)

__all__ = [
    "ROS2_AVAILABLE",
    "TOPIC_TYPES",
    "require_ros2",
    "frame_to_dict",
    "dict_to_frame",
    "map_to_dict",
    "answer_to_dict",
    "camera_info_to_intrinsics",
]
