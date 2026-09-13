# -*- coding: utf-8 -*-
"""文档数字的回归锁：文档里写的规模数字必须与代码实测一致。

为什么把它做成测试而不是"记得去核对"
==================================
这个项目已经**两次**出现"文档里的数字是错的"：

1. 新建清单文档时**凭印象写模块行数** —— 29 处全错；
2. 加了 48 个测试后，**12 处文档还写着"386 个测试"**；
3. 再往后我又手写了 5 个测试文件行数 —— 又错了 5 处。

三次都是同一个模式：**数字看起来精确、排版整齐，读者会默认它是量出来的**。
所以必须机器守。红了怎么办（有明确修法）：

    python scripts/25_audit_doc_numbers.py --fix-lines     # 修正表格里的行数
    python scripts/25_audit_doc_numbers.py --fix           # 修正测试总数

本文件只断言**稳定且重要**的几类：
  · 测试总数 / 默认数 / 慢速数（任何文档只要提到，就必须对）
  · 全局规模表（源码/测试/脚本/文档 的文件数与行数）
  · 按包分布表

**刻意不做**的事：断言实验指标（F1、误差、吞吐…）。那些数字来自跑批产物，
不是从代码里能算出来的，用测试锁会变成"改数据就红"的噪声。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [ROOT / "README.md", ROOT / "AGENTS.md"]
DOC_FILES += sorted((ROOT / "docs").glob("*.md"))
DOC_FILES += [ROOT / "ros2_ws" / "README.md",
              ROOT / "src" / "roboground" / "deployment" / "ros2" / "README.md"]

FIX_HINT = (
    "\n修法（不要手改，行数和测试数都是机器能算的）：\n"
    "  python scripts/25_audit_doc_numbers.py --fix-lines   # 表格行数\n"
    "  python scripts/25_audit_doc_numbers.py --fix         # 测试总数\n"
    "  再跑 python scripts/25_audit_doc_numbers.py 复查（应当 0 处待确认）\n"
)


def _collect_tests(extra: list) -> int:
    """用 pytest 自己数测试。`-o addopts=` 是必须的，见下面注释。"""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q",
           "--no-header", "-p", "no:cacheprovider", "-o", "addopts=", *extra]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    text = out.stdout + out.stderr
    m = re.search(r"(\d+)/\d+ tests collected", text) or \
        re.search(r"(\d+) tests collected", text)
    assert m, f"数不出测试数量，pytest 输出：\n{text[-800:]}"
    return int(m.group(1))


@pytest.fixture(scope="module")
def real_counts():
    total = _collect_tests([])
    slow = _collect_tests(["-m", "slow"])
    return {"total": total, "slow": slow, "default": total - slow}


def _python_scale(base: Path):
    files = [p for p in base.rglob("*.py") if "__pycache__" not in p.parts]
    lines = sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines())
                for p in files)
    return len(files), lines


def test_docs_do_not_claim_wrong_test_counts(real_counts):
    """任何文档里的"N 个测试"都必须等于真实测试数（或默认数/慢速数）。

    这条守的是最容易被引用、也最容易过期的数字 —— 简历和面试稿里都在写它。
    """
    allowed = {real_counts["total"], real_counts["default"], real_counts["slow"]}
    offenders = []
    for f in DOC_FILES:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for num in re.findall(r"(\d{2,4})\s*个测试", line):
                n = int(num)
                if n > 100 and n not in allowed:
                    offenders.append(f"{f.relative_to(ROOT)}:{i} 写了「{num} 个测试」，"
                                     f"实际 total={real_counts['total']} / "
                                     f"default={real_counts['default']} / "
                                     f"slow={real_counts['slow']}")
            # `**386**（362 默认 + 24 慢速）` 这种没有"个测试"字样的写法
            for mm in re.finditer(r"\*\*(\d{3,4})\s*个?\*\*（(\d{3,4}) 默认 \+ (\d{1,3})", line):
                if (int(mm.group(1)), int(mm.group(2)), int(mm.group(3))) != (
                        real_counts["total"], real_counts["default"], real_counts["slow"]):
                    offenders.append(f"{f.relative_to(ROOT)}:{i} 写了 "
                                     f"{mm.group(1)}/{mm.group(2)}/{mm.group(3)}，"
                                     f"实际 {real_counts['total']}/"
                                     f"{real_counts['default']}/{real_counts['slow']}")
    assert not offenders, "文档里的测试数与实际不符：\n" + "\n".join(
        f"  {o}" for o in offenders) + FIX_HINT


def test_project_scale_table_matches_reality():
    """`docs/项目完整清单与阅读顺序.md` 的全局规模表必须是实测值。"""
    doc = ROOT / "docs" / "项目完整清单与阅读顺序.md"
    if not doc.exists():
        pytest.skip("没有清单文档")
    text = doc.read_text(encoding="utf-8", errors="replace")
    expected = {
        "源码": _python_scale(ROOT / "src"),
        "测试": _python_scale(ROOT / "tests"),
    }
    scripts = [p for p in (ROOT / "scripts").rglob("*")
               if p.is_file() and p.suffix in (".py", ".sh", ".ps1")]
    expected["脚本"] = (len(scripts),
                      sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines())
                          for p in scripts))
    docs = sorted((ROOT / "docs").glob("*.md"))
    expected["文档"] = (len(docs),
                      sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines())
                          for p in docs))

    offenders = []
    for key, (files, lines) in expected.items():
        mm = re.search(rf"\|\s*{key}[^|]*\|\s*\*{{0,2}}([\d,]+)\*{{0,2}}\s*\|\s*"
                       rf"\*{{0,2}}([\d,]+)", text)
        if not mm:
            offenders.append(f"找不到「{key}」那一行")
            continue
        cf, cl = int(mm.group(1).replace(",", "")), int(mm.group(2).replace(",", ""))
        if (cf, cl) != (files, lines):
            offenders.append(f"「{key}」写了 {cf} 文件 / {cl} 行，实际 {files} 文件 / {lines} 行")
    assert not offenders, "清单文档的全局规模表与实际不符：\n" + "\n".join(
        f"  {o}" for o in offenders) + FIX_HINT


def test_per_package_table_matches_reality():
    """清单文档的「按包分布」表必须是实测值（行数占比是面试要讲的论据）。"""
    doc = ROOT / "docs" / "项目完整清单与阅读顺序.md"
    if not doc.exists():
        pytest.skip("没有清单文档")
    text = doc.read_text(encoding="utf-8", errors="replace")
    base = ROOT / "src" / "roboground"
    real = {}
    root_files = list(base.glob("*.py"))
    real["根模块"] = (len(root_files),
                    sum(len(f.read_text(encoding="utf-8", errors="replace").splitlines())
                        for f in root_files))
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "__pycache__"):
        fs = [f for f in d.rglob("*.py") if "__pycache__" not in f.parts]
        real[d.name + "/"] = (len(fs),
                            sum(len(f.read_text(encoding="utf-8", errors="replace").splitlines())
                                for f in fs))

    offenders = []
    for pkg, (files, lines) in real.items():
        mm = re.search(rf"\|\s*`?{re.escape(pkg)}`?\s*\|\s*(\d+)\s*\|\s*\*{{0,2}}([\d,]+)",
                       text)
        if not mm:
            offenders.append(f"找不到「{pkg}」那一行")
            continue
        cf, cl = int(mm.group(1)), int(mm.group(2).replace(",", ""))
        if (cf, cl) != (files, lines):
            offenders.append(f"「{pkg}」写了 {cf} 文件 / {cl} 行，实际 {files} 文件 / {lines} 行")
    assert not offenders, "按包分布表与实际不符：\n" + "\n".join(
        f"  {o}" for o in offenders) + FIX_HINT


# ==========================================================================
# 跨文档的**全局计数**（这类数字之前被审计脚本漏掉了，靠人工发现）
# ==========================================================================
def test_docs_global_counts_match_reality():
    """★ 任何文档里的「docs N 篇 / scripts N 个 / src N 文件 N 行 / tests N 文件」
    都必须与实际一致。

    为什么单列一条：`scripts/25_audit_doc_numbers.py` 只覆盖
    「清单文档」里的规模表与按包表，**管不到散落在其它文档里的同类说法** ——
    实测漏过 13 处（`docs/` 篇数在 README 与清单里写 15、实际 18；
    `scripts` 写 31、实际 36；`data/` 行数在 3 个文件里写 6,306、实际 7,324；
    `geometry/` 写 1,136、实际 1,240；"445 全绿"等）。
    这类数字**最容易过期又最显眼**（简历、面试稿都在引用），所以自动守。
    """
    import pathlib

    def _py(base: pathlib.Path):
        fs = [p for p in base.rglob("*.py") if "__pycache__" not in p.parts]
        return len(fs), sum(len(p.read_text(encoding="utf-8", errors="replace")
                                .splitlines()) for p in fs)

    n_docs = len(list((ROOT / "docs").glob("*.md")))
    scr = [p for p in (ROOT / "scripts").glob("*.py")]
    n_scr = len([p for p in scr if p.name[:2].isdigit()])
    n_src, l_src = _py(ROOT / "src")
    n_tst, l_tst = _py(ROOT / "tests")
    pkg = {}
    for d in sorted(p for p in (ROOT / "src" / "roboground").iterdir()
                    if p.is_dir() and p.name != "__pycache__"):
        fs = [f for f in d.rglob("*.py") if "__pycache__" not in f.parts]
        pkg[d.name] = (len(fs),
                       sum(len(f.read_text(encoding="utf-8", errors="replace").splitlines())
                           for f in fs))

    allowed = {
        "docs 篇数": {n_docs},
        "scripts 个数": {n_scr},
        "src 文件数": {n_src},
        "tests 文件数": {n_tst},
    }
    # (正则, 取第几个捕获组作为该计数)
    # ⚠️ 必须**显式指定组号**：同一行里同时有「测试数」和「文件数」
    #   （如 `tests **516 个**（23 文件 7,941 行）`），
    #   若把所有捕获组都当候选值，会把 516 误判成"文件数"而误报。
    #   第一版就是这么写的，直接产生了一条假告警。
    patterns = [
        ("docs 篇数", r"docs/?\s*[`\s]*(\d+)\s*篇", 1),
        ("docs 篇数", r"文档清单（(\d+)\s*份", 1),
        ("scripts 个数", r"scripts/\*?\.?p?y?`?\s*[（(](\d+)\s*个", 1),
        ("src 文件数", r"src\s*\*{0,2}(\d+)\s*文件", 1),
        ("tests 文件数", r"tests\s*\*{0,2}\d+\s*个\*{0,2}（(\d+)\s*文件", 1),
    ]
    offenders = []
    for f in DOC_FILES:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            for name, pat, grp in patterns:
                for m in re.finditer(pat, line):
                    v = int(m.group(grp))
                    if v not in allowed[name]:
                        offenders.append(
                            f"{f.relative_to(ROOT)}:{i} 写了「{name} = {v}」，"
                            f"实际 {sorted(allowed[name])}")
            # 按包行数：任何文档里出现 `data/` / `geometry/` 的「N 文件 / N 行」
            # 或表格形式 `| \`data/\` | N | N |`
            for p, (files, lines) in pkg.items():
                for m in re.finditer(
                        rf"`{re.escape(p)}/?`[^|\n]*\|\s*\*{{0,2}}(\d+)\*{{0,2}}\s*\|\s*\*{{0,2}}([\d,]+)",
                        line):
                    cf, cl = int(m.group(1)), int(m.group(2).replace(",", ""))
                    if (cf, cl) != (files, lines):
                        offenders.append(
                            f"{f.relative_to(ROOT)}:{i} 里 `{p}/` 写了 {cf} 文件 / "
                            f"{cl} 行，实际 {files} 文件 / {lines} 行")

    assert not offenders, "跨文档的全局计数与实际不符：\n" + "\n".join(
        f"  {o}" for o in dict.fromkeys(offenders)) + FIX_HINT
