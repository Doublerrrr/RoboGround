#!/usr/bin/env bash
# 诊断：WSL 里 pip / setuptools 的版本与可用的安装方式
set +u
source /opt/ros/humble/setup.bash >/dev/null 2>&1
set -u
echo "--- python / pip ---"
python3 -V
python3 -m pip -V
echo
echo "--- setuptools / wheel ---"
python3 -c "import setuptools; print('setuptools', setuptools.__version__)" 2>&1
python3 -c "import wheel; print('wheel', wheel.__version__)" 2>&1
echo
echo "--- pip 配置的 index ---"
python3 -m pip config list 2>&1
cat /etc/pip.conf 2>/dev/null || echo "(无 /etc/pip.conf)"
cat ~/.pip/pip.conf 2>/dev/null || echo "(无 ~/.pip/pip.conf)"
cat ~/.config/pip/pip.conf 2>/dev/null || echo "(无 ~/.config/pip/pip.conf)"
echo
echo "--- 网络：能否访问 pypi 镜像 ---"
timeout 12 python3 -c "
import urllib.request
for url in ('https://pypi.tuna.tsinghua.edu.cn/simple/', 'https://pypi.org/simple/'):
    try:
        with urllib.request.urlopen(url, timeout=6) as r:
            print(f'  OK  {url} -> {r.status}')
    except Exception as exc:
        print(f'  BAD {url} -> {type(exc).__name__}: {exc}')
" 2>&1
echo
echo "--- 已安装的 roboground（若有）---"
python3 -c "import roboground, sys; print(roboground.__file__)" 2>&1 | tail -1
python3 -m pip show roboground 2>&1 | head -8
echo
echo "--- 非 editable 安装是否可行（只做 dry-run 检查构建后端）---"
python3 -c "
import setuptools.build_meta as bm
print('build_wheel   :', hasattr(bm, 'build_wheel'))
print('build_editable:', hasattr(bm, 'build_editable'))
" 2>&1
