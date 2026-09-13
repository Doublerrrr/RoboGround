#!/usr/bin/env bash
# 44 · 诊断：message_filters.registerCallback 是"替换"还是"追加"？
#
# 这个语义直接决定"打点包装回调"能不能用来做延迟测量：
# 如果是**追加**，那么包装一次就等于每帧被处理两遍，
# 测出来的耗时会被整体放大一倍（而且看起来还挺"合理"，极难发现）。
set +u
source /opt/ros/humble/setup.bash >/dev/null 2>&1
set -u

python3 - <<'PY'
import inspect

import message_filters
from message_filters import SimpleFilter, TimeSynchronizer

print("message_filters:", message_filters.__file__)
print()
print("--- SimpleFilter.registerCallback ---")
try:
    print(inspect.getsource(SimpleFilter.registerCallback))
except Exception as exc:
    print("(取不到源码)", exc)

has_own = "registerCallback" in TimeSynchronizer.__dict__
print("TimeSynchronizer 自己实现了 registerCallback？", has_own)
if has_own:
    print(inspect.getsource(TimeSynchronizer.registerCallback))

# 直接实测：注册两个回调，看是替换还是追加
class Fake(SimpleFilter):
    def __init__(self):
        super().__init__()
        self.hits = []

    def add(self, msg):
        pass


f = Fake()
calls = []
f.registerCallback(lambda *a: calls.append("A"))
f.registerCallback(lambda *a: calls.append("B"))
attrs = {k: v for k, v in vars(f).items() if "call" in k.lower()}
print("\n注册两次后，实例上的 callback 相关属性：")
for k, v in attrs.items():
    print(f"  {k} = {v!r}")

# 触发一次，看有几个回调被调用
import types
try:
    # 不同版本里字段名可能是 callback / callbacks
    if hasattr(f, "callback") and f.callback is not None:
        f.callback(None)
    for cb in getattr(f, "callbacks", []) or []:
        cb(None)
except Exception as exc:
    print("触发时出错：", exc)
print("触发一次后调用记录 =", calls)
print()
if calls == ["B"]:
    print("结论：registerCallback 是**替换** —— 包装打点是安全的")
elif calls == ["A", "B"]:
    print("结论：registerCallback 是**追加** —— 包装打点会让每帧被处理两遍！")
else:
    print("结论：未能确定，请人工看上面的属性")
PY
