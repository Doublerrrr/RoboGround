#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""01 · 环境自检（等价于 `roboground check`，但不依赖包安装）。

用法::

    python scripts/01_check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.cli import cmd_check, build_parser  # noqa: E402


def main() -> int:
    parser = build_parser()
    args = parser.parse_args(["check"])
    return cmd_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
