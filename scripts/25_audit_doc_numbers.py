#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""25 · 文档数字审计：把文档里写的数字和**代码实测**对一遍。

为什么需要它
==========
这个项目已经出过两次"文档里的数字是错的"：

1. 新增文档时**凭印象写模块行数** —— 29 处行数是编的，被脚本抓出来才改对；
2. 加了 48 个测试后，**12 处文档还写着"386 个测试"**。

而且这两类错误都**极其隐蔽**：数字看起来精确、排版整齐，
读者（包括面试官）会默认它是量出来的。所以必须机器对，不能靠人眼。

做三件事
=======
1. **实测**：跑 `pytest --collect-only` 拿测试数与分文件明细；
   统计 `src/`、`tests/`、`scripts/` 的文件数与行数；数文档篇数。
2. **对账**：扫描所有 Markdown，找出"测试数""行数"之类的断言，与实测比对。
3. **可选修复**：`--fix` 只做**白名单内的精确替换**（例如 `386 个测试` → `434 个测试`），
   绝不使用模糊正则批量改数字 —— 那会把"10 个测试类别"这种无关数字也改掉。

用法::

    python scripts/25_audit_doc_numbers.py          # 只报告
    python scripts/25_audit_doc_numbers.py --fix    # 报告并修复白名单项
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "setup_env.md"]
DOC_FILES += sorted((ROOT / "docs").glob("*.md"))
DOC_FILES += [ROOT / "ros2_ws" / "README.md",
              ROOT / "src" / "roboground" / "deployment" / "ros2" / "README.md"]


# --------------------------------------------------------------------------
# 实测
# --------------------------------------------------------------------------
def measure_tests() -> Dict:
    """用 pytest 自己数，避免手数漏掉参数化用例。"""
    def collect(extra: List[str]) -> Tuple[int, Dict[str, int]]:
        # ⚠️ 必须用 `-o addopts=` 清掉 pyproject 里的 `-q`：
        #    `-q` 再加自己的 `-q` = `-qq`，会把末尾的
        #    "434/434 tests collected" 汇总行**整行去掉**，于是数出来是 0。
        #    （本项目在别处也踩过这个坑：`-qq` 会让结论看起来"什么都没跑"。）
        cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q",
               "--no-header", "-p", "no:cacheprovider", "-o", "addopts=", *extra]
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        text = out.stdout + out.stderr
        total = 0
        m = re.search(r"(\d+)/\d+ tests collected", text) or \
            re.search(r"(\d+) tests collected", text)
        if m:
            total = int(m.group(1))
        per_file: Dict[str, int] = {}
        for line in text.splitlines():
            mm = re.match(r"(tests[\\/][\w\\/]+\.py)::", line.strip())
            if mm:
                key = mm.group(1).replace("\\", "/").split("/")[-1]
                per_file[key] = per_file.get(key, 0) + 1
        return total, per_file

    all_total, per_file_all = collect([])
    slow_total, _ = collect(["-m", "slow"])
    # `-m slow` 会覆盖 addopts 里的 `-m 'not slow'`；默认数 = 总数 - slow
    return {"total": all_total, "slow": slow_total,
            "default": all_total - slow_total, "per_file": per_file_all}


def count_python(root: Path) -> Tuple[int, int]:
    files = [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
    lines = 0
    for p in files:
        lines += len(p.read_text(encoding="utf-8", errors="replace").splitlines())
    return len(files), lines


def measure_tree() -> Dict:
    src_files, src_lines = count_python(ROOT / "src")
    test_files, test_lines = count_python(ROOT / "tests")
    scripts = [p for p in (ROOT / "scripts").rglob("*")
               if p.is_file() and p.suffix in (".py", ".sh", ".ps1")]
    script_lines = sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines())
                       for p in scripts)
    docs = sorted((ROOT / "docs").glob("*.md"))
    doc_lines = sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines())
                    for p in docs)
    return {
        "src_files": src_files, "src_lines": src_lines,
        "test_files": test_files, "test_lines": test_lines,
        "scripts": len(scripts), "script_lines": script_lines,
        "docs": len(docs), "doc_lines": doc_lines,
    }


def package_line_counts() -> Dict[str, Tuple[int, int]]:
    """{包名: (文件数, 行数)}，用于核对"全局规模"里的按包分布表。"""
    base = ROOT / "src" / "roboground"
    out: Dict[str, Tuple[int, int]] = {}
    root_files = [f for f in base.glob("*.py")]
    out["根模块"] = (len(root_files),
                   sum(len(f.read_text(encoding="utf-8", errors="replace").splitlines())
                       for f in root_files))
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "__pycache__"):
        fs = [f for f in d.rglob("*.py") if "__pycache__" not in f.parts]
        out[d.name + "/"] = (len(fs),
                            sum(len(f.read_text(encoding="utf-8", errors="replace").splitlines())
                                for f in fs))
    return out


def module_line_counts() -> Dict[str, int]:
    """{模块文件名: 行数}，用于核对文档表格里的"行数"列。

    ⚠️ 按**文件名**索引会有歧义（项目里有 `deployment/pipeline.py` 与
    `data/video/pipeline.py` 两个同名文件）。这里遇到重名时记为
    "歧义"（值取 -1），让审计报出来而不是随便挑一个 ——
    否则会拿错文件的行数去"纠正"文档，越改越错。
    """
    counts: Dict[str, List[int]] = {}
    for base in (ROOT / "src", ROOT / "tests"):
        for p in base.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            n = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
            counts.setdefault(p.name, []).append(n)
    return {k: (v[0] if len(v) == 1 else -1) for k, v in counts.items()}


# --------------------------------------------------------------------------
# 对账
# --------------------------------------------------------------------------
#: 具体数字 → 该数字的正确值（精确字符串替换，避免误伤无关数字）
COUNT_FIXES = {
    "386 个测试": "{total} 个测试",
    "386 个默认测试": "{default} 个默认测试",
    "386 个测试（362 默认": "{total} 个测试（{default} 默认",
    "386 个测试全绿（362 默认": "{total} 个测试全绿（{default} 默认",
    "362 个默认测试": "{default} 个默认测试",
    "362 默认 + 24": "{default} 默认 + {slow}",
    "共 386，含 24 个 slow": "共 {total}，含 {slow} 个 slow",
    "410 个默认测试（离线": "{default} 个默认测试（离线",
    "| 测试数量           | **386**（362 默认 + 24 慢速真实数据），全绿 |":
        "| 测试数量           | **{total}**（{default} 默认 + {slow} 慢速真实数据），全绿 |",
    "**386**（362 默认 + 24 真实数据）": "**{total}**（{default} 默认 + {slow} 真实数据）",
    "**386**（362 默认 + 24 慢速），全绿":
        "**{total}**（{default} 默认 + {slow} 慢速），全绿",
    "**386 个测试**": "**{total} 个测试**",
    "**386 个**（362 默认 + 24 慢速）": "**{total} 个**（{default} 默认 + {slow} 慢速）",
    "386 全绿（362 默认 + 24 真实数据）":
        "{total} 全绿（{default} 默认 + {slow} 真实数据）",
}


def scan_claims(m: Dict, tree: Dict, mods: Dict[str, int], pkgs: Dict):
    """返回 (需要修的替换清单, 供人看的告警清单)。"""
    fixable: List[Tuple[Path, str, str]] = []
    warnings: List[Tuple[Path, int, str]] = []
    fmt = {"total": m["total"], "default": m["default"], "slow": m["slow"]}

    for f in DOC_FILES:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")

        # 1) 白名单里的精确串 → 可自动修
        for old, new_tpl in COUNT_FIXES.items():
            if old in text:
                new = new_tpl.format(**fmt)
                if old != new:
                    fixable.append((f, old, new))

        # 1b) ★ 模式化的总测试数修正（含**形容词**的写法）
        #
        # 为什么需要：`COUNT_FIXES` 是**硬编码白名单**，键会随数字更新而失效 ——
        # 实测就出现过"白名单里全是 386，而文档早已改成 494"的情况，
        # 于是 `--fix` **静默什么都不做**，数字继续漂。
        #
        # 为什么仍然安全（三道护栏，缺一不可）：
        #   · 只改 **3 位数且 > 100** 的数字 —— 测试总数是三位数；
        #     单文件计数（十几~几十）与"10 个测试类别"这类都不会被碰；
        #   · **跳过含具体测试文件路径的行**（`tests/test_x.py`）——
        #     那些是单文件计数，由另一条规则负责；
        #   · **跳过含子集限定词的行**（相关/其中/这类…）——
        #     "test_deployment.py 里有 10 个相关测试"说的是子集，不是总数。
        _file_ref = re.compile(r"tests[\\/][\w\\/]+\.py")
        _subset = ("相关", "其中", "这类", "此类", "那些", "上述", "涉及", "部分")
        _claim = re.compile(r"(\d{3,4})\s*个([^，。、\s]{0,6})(测试)")
        for line in text.splitlines():
            if _file_ref.search(line):
                continue
            for mm in _claim.finditer(line):
                n, adj = int(mm.group(1)), mm.group(2)
                if n <= 100 or any(w in adj for w in _subset):
                    continue
                if n in (m["total"], m["default"], m["slow"]):
                    continue
                # 形容词决定该替换成总数还是默认数
                target = m["default"] if "默认" in adj else m["total"]
                old_s, new_s = mm.group(0), f"{target} 个{adj}测试"
                if old_s != new_s:
                    fixable.append((f, old_s, new_s))

        for i, line in enumerate(text.splitlines(), 1):
            # 2) 任何"NNN 个[形容词]测试"里的 NNN 与实际不符 → 告警
            for mm in re.finditer(r"(\d{2,4})\s*个[^，。、\s]{0,6}测试", line):
                num = mm.group(1)
                n = int(num)
                if (_file_ref.search(line) or n <= 100
                        or any(w in mm.group(0) for w in _subset)):
                    continue
                if n not in (m["total"], m["slow"], m["default"]):
                    warnings.append((f, i, f"「{num} 个测试」与实际（{m['total']}）不符"))
            # 3) 分文件测试数：同一行里出现 test_xxx.py 和 N 个测试
            files_in_line = re.findall(r"(test_\w+\.py)", line)
            nums_in_line = [int(x) for x in re.findall(r"(\d+)\s*个测试", line)]
            if files_in_line and nums_in_line:
                for fname in files_in_line:
                    real = m["per_file"].get(fname)
                    if real is not None and nums_in_line[0] != real:
                        warnings.append(
                            (f, i, f"{fname} 的测试数：文档写 {nums_in_line[0]}，实际 {real}"))
            # 4) 表格里的模块行数：| `x.py` | 123 | ... |
            for mm in re.finditer(r"\|\s*`([\w./]+\.py)`\s*\|\s*(\d{2,5})\s*\|", line):
                name, claimed = mm.group(1).split("/")[-1], int(mm.group(2))
                real = mods.get(name)
                if real is None or real < 0:
                    continue          # 未知文件或重名歧义 → 不瞎报
                if claimed != real:
                    warnings.append((f, i, f"{name} 行数：文档写 {claimed}，实际 {real}"))

            # 5) 按包分布：| `data/` | 18 | **6,306** | ...
            for mm in re.finditer(
                    r"\|\s*`?([\w]+/|根模块)`?\s*\|\s*(\d+)\s*\|\s*\*{0,2}([\d,]+)\*{0,2}\s*\|", line):
                pkg = mm.group(1)
                if not pkg.endswith("/"):
                    pkg = "根模块"
                claimed_files = int(mm.group(2))
                claimed_lines = int(mm.group(3).replace(",", ""))
                real = pkgs.get(pkg)
                if real is None:
                    continue
                if (claimed_files, claimed_lines) != real:
                    warnings.append((f, i,
                                     f"{pkg} 规模：文档写 {claimed_files} 文件/{claimed_lines} 行，"
                                     f"实际 {real[0]} 文件/{real[1]} 行"))

            # 6) 全局规模表：| 源码 `src/roboground/` | **71** | **19,884** |
            for key, real_files, real_lines in (
                ("源码", tree["src_files"], tree["src_lines"]),
                ("测试", tree["test_files"], tree["test_lines"]),
                ("脚本", tree["scripts"], tree["script_lines"]),
                ("文档", tree["docs"], tree["doc_lines"]),
            ):
                mm = re.search(
                    rf"\|\s*{key}[^|]*\|\s*\*{{0,2}}([\d,]+)\*{{0,2}}\s*\|\s*\*{{0,2}}([\d,]+)",
                    line)
                if mm:
                    cf, cl = int(mm.group(1).replace(",", "")), int(mm.group(2).replace(",", ""))
                    if key == "测试" and cf == m["total"]:
                        continue          # "测试 442 个" 那一行单独判
                    if (cf, cl) != (real_files, real_lines):
                        warnings.append((f, i,
                                         f"{key} 规模：文档写 {cf} 文件/{cl} 行，"
                                         f"实际 {real_files} 文件/{real_lines} 行"))
    return fixable, warnings


def apply_fixes(fixable, fmt: Dict) -> int:
    by_file: Dict[Path, List[Tuple[str, str]]] = {}
    for f, old, new in fixable:
        by_file.setdefault(f, []).append((old, new))
    n = 0
    for f, pairs in by_file.items():
        text = f.read_text(encoding="utf-8", errors="replace")
        for old, new in pairs:
            c = text.count(old)
            text = text.replace(old, new)
            n += c
            print(f"  [fix] {f.relative_to(ROOT)}: {c}× 「{old}」 → 「{new}」")
        f.write_text(text, encoding="utf-8")
    return n


def fix_module_lines(mods: Dict[str, int]) -> int:
    """把表格里的模块行数**批量**改成实测值。

    为什么值得单独一个自动修复：本项目已经**两次**在文档里手写行数写错
    （一次 29 处全错、一次 5 处错）。行数是"改一次代码就过期"的数字，
    靠人维护必然出错；能机器改的就别手改。

    只动形如 ``| `x.py` | 123 | ...`` 的表格行，且只改那个纯数字单元格。
    """
    changed = 0
    for f in DOC_FILES:
        if not f.exists():
            continue
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        out = []
        touched = 0
        for line in lines:
            mm = re.search(r"\|\s*`([\w./]+\.py)`\s*\|\s*(\d{2,5})\s*\|", line)
            if mm:
                name = mm.group(1).split("/")[-1]
                real = mods.get(name)
                if real is not None and real > 0 and int(mm.group(2)) != real:
                    line = line[:mm.start(2)] + str(real) + line[mm.end(2):]
                    touched += 1
                    changed += 1
            out.append(line)
        if touched:
            f.write_text("".join(out), encoding="utf-8")
            print(f"  [fix-lines] {f.relative_to(ROOT)}: {touched} 处行数已改为实测值")
    return changed


def fix_scale_rows(tree: Dict, m: Dict) -> int:
    """修正"全局规模"表里的 源码/测试/脚本/文档 四个数字。

    为什么连聚合行也自动改：这些数字**每加一个文件就过期**，
    而它们恰好是最常被引用、最常进面试稿的（"71 个文件 / 19,884 行 / 442 个测试"）。
    手改必错 —— 本项目已经错过两次。所以 `--fix-lines` 一次把
    模块行数与聚合数字都修掉。
    """
    doc = ROOT / "docs" / "项目完整清单与阅读顺序.md"
    if not doc.exists():
        return 0
    targets = {
        "源码": (tree["src_files"], tree["src_lines"]),
        "测试": (tree["test_files"], tree["test_lines"]),
        "脚本": (tree["scripts"], tree["script_lines"]),
        "文档": (tree["docs"], tree["doc_lines"]),
    }
    lines = doc.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    out, n = [], 0
    for line in lines:
        for key, (files, total_lines) in targets.items():
            mm = re.search(rf"(\|\s*{key}[^|]*\|\s*\*{{0,2}})([\d,]+)(\*{{0,2}}\s*\|\s*"
                           rf"\*{{0,2}})([\d,]+)", line)
            if not mm:
                continue
            if (int(mm.group(2).replace(",", "")), int(mm.group(4).replace(",", ""))) \
                    != (files, total_lines):
                line = (line[:mm.start(2)] + f"{files:,}" + line[mm.end(2):mm.start(4)]
                        + f"{total_lines:,}" + line[mm.end(4):])
                n += 1
        out.append(line)
    if n:
        doc.write_text("".join(out), encoding="utf-8")
        print(f"  [fix-scale] {doc.relative_to(ROOT)}: {n} 处聚合数字已更新")
    return n


def fix_generic_counts(m: Dict) -> int:
    """把任何"N 个测试"里的错误 N 改成真实值（不依赖白名单）。

    为什么改成通用替换：白名单只覆盖"我当时知道的那几个错值"，
    下次再加测试就又漏了（已经发生过：386 → 442 之后，白名单还是旧的）。
    通用规则更省心：

      · 只动 **3~4 位数**且 **> 100** 的数字 → 不会误伤"10 个测试类别"；
      · **跳过点名 `test_xxx.py` 的行** → 那种是"单文件测试数"，
        不能改成总数（会被 3 位数的单文件数误伤）；
      · 允许值 = {总数, 默认数, 慢速数}，其余一律替换成**总数**。

    另外处理 `**N**（A 默认 + B 慢速…）` 这种三数并排的写法。
    """
    allowed = {m["total"], m["default"], m["slow"]}
    n = 0
    for f in DOC_FILES:
        if not f.exists():
            continue
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        out = []
        for line in lines:
            if re.search(r"test_\w+\.py", line):
                out.append(line)          # 单文件测试数，不碰
                continue
            orig = line

            def _sub(mm):
                return f"{m['total']} 个测试" if int(mm.group(1)) not in allowed \
                    else mm.group(0)

            line = re.sub(r"(\d{3,4})\s*个测试", _sub, line)

            def _sub3(mm):
                nums = (int(mm.group(1)), int(mm.group(2)), int(mm.group(3)))
                if nums == (m["total"], m["default"], m["slow"]):
                    return mm.group(0)
                return (f"**{m['total']}**（{m['default']} 默认 + {m['slow']}")

            line = re.sub(r"\*\*(\d{3,4})\s*个?\*\*（(\d{3,4}) 默认 \+ (\d{1,3})", _sub3, line)

            def _sub2(mm):
                nums = (int(mm.group(1)), int(mm.group(2)))
                if nums == (m["default"], m["slow"]):
                    return mm.group(0)
                return f"{m['default']} 默认 + {m['slow']}"

            line = re.sub(r"(\d{3,4}) 默认 \+ (\d{1,3})", _sub2, line)
            if line != orig:
                n += 1
            out.append(line)
        f.write_text("".join(out), encoding="utf-8")
    if n:
        print(f"  [fix-counts] {n} 行里的测试数字已更新")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true", help="应用白名单内的精确修复")
    ap.add_argument("--fix-lines", action="store_true",
                    help="把表格里的模块行数批量改成实测值（推荐，别手改）")
    args = ap.parse_args()

    print("=" * 84)
    print("文档数字审计")
    print("=" * 84)

    print("\n[实测]")
    m = measure_tests()
    tree = measure_tree()
    mods = module_line_counts()
    pkgs = package_line_counts()
    print(f"  测试：总 {m['total']} = 默认 {m['default']} + slow {m['slow']}")
    print(f"  src    : {tree['src_files']} 文件 / {tree['src_lines']} 行")
    print(f"  tests  : {tree['test_files']} 文件 / {tree['test_lines']} 行")
    print(f"  scripts: {tree['scripts']} 个 / {tree['script_lines']} 行")
    print(f"  docs   : {tree['docs']} 篇 / {tree['doc_lines']} 行")

    fixable, warnings = scan_claims(m, tree, mods, pkgs)

    if args.fix_lines:
        print("\n[批量修正表格行数]")
        n_lines = fix_module_lines(mods) + fix_scale_rows(tree, m)
        print(f"  共 {n_lines} 处")
        fixable, warnings = scan_claims(m, tree, mods, pkgs)

    print(f"\n[可自动修复的白名单项] {len(fixable)} 处")
    for f, old, new in fixable:
        print(f"  {f.relative_to(ROOT)}: 「{old}」 → 「{new}」")

    print(f"\n[需要人工确认] {len(warnings)} 处")
    for f, i, msg in warnings:
        print(f"  {f.relative_to(ROOT)}:{i}  {msg}")

    # ⚠️ 条件必须是 `if args.fix:` 而不是 `if args.fix and fixable:` ——
    # 白名单命中为空时也要跑**通用**修复（白名单只覆盖已知的几个旧错值，
    # 下一次加测试就又会漏，通用替换才是主力）。
    if args.fix:
        fmt = {"total": m["total"], "default": m["default"], "slow": m["slow"]}
        print("\n[应用修复]")
        n = apply_fixes(fixable, fmt) if fixable else 0
        n += fix_generic_counts(m)
        print(f"  共替换 {n} 处")
        # 复查
        fixable2, warnings2 = scan_claims(m, tree, mods, pkgs)
        print(f"\n[复查] 剩余可修项 {len(fixable2)}，待人工确认 {len(warnings2)}")
        return 0 if not fixable2 else 1

    if not fixable and not warnings:
        print("\n[OK] 文档里的数字与代码实测一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
