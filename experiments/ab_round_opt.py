"""端到端配对 A/B：按 ON,OFF,OFF,ON 顺序跑 1080M 基准，抵消热漂移。

单次 1080M 基准约 50s，连跑十几分钟后 GPU 降频会让 fwd/bwd 整体慢 20%，
所以「先测改动前、再测改动后」的顺序比较会把降频算到后测的一方头上，
两次测量的差可以完全被漂移淹没。对称顺序下线性漂移对两组影响相同，
配对比较才有意义；min 口径比中位更抗优化器 NS 刷新造成的周期尖峰。

被测改动要先用环境变量做成可开关的（ON=改动生效）。例如本轮：

    uv run python experiments/ab_round_opt.py VIBY_SITU_PACKED=1/0

多个开关用空格分隔，同开同关。

用法: uv run python experiments/ab_round_opt.py VAR=on/off [...] [--iters N] [--cool S]
"""

import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(
    r"归一化单微批: ([\d.]+)ms（中位） / ([\d.]+)ms（周期均值） / ([\d.]+)ms"
)

iters, cool, flags = 12, 60, {}
argv = sys.argv[1:]
i = 0
while i < len(argv):
    a = argv[i]
    if a == "--iters":
        i += 1
        iters = int(argv[i])
    elif a == "--cool":
        i += 1
        cool = int(argv[i])
    else:
        name, vals = a.split("=", 1)
        on, off = vals.split("/", 1)
        flags[name] = (on, off)
    i += 1
if not flags:
    raise SystemExit(__doc__)


def run(on: bool):
    env = dict(os.environ)
    for name, (v_on, v_off) in flags.items():
        env[name] = v_on if on else v_off
    out = subprocess.run(
        [
            sys.executable,
            "experiments/bench_train_step.py",
            str(iters),
            "--preset",
            "1080m",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    ).stdout
    m = PAT.search(out)
    if not m:
        print(out[-2000:])
        raise SystemExit("未解析到基准结果")
    return tuple(float(g) for g in m.groups())  # (中位, 周期均值, min)


print(
    f"开关: {', '.join(f'{k}={v[0]}/{v[1]}' for k, v in flags.items())}"
    f"  iters={iters} cooldown={cool}s\n"
)
order = [True, False, False, True]
res = {True: [], False: []}
for i, on in enumerate(order):
    if i:
        time.sleep(cool)
    med, avg, mn = run(on)
    res[on].append((med, avg, mn))
    print(
        f"[{i + 1}/{len(order)}] {'ON ' if on else 'OFF'}  "
        f"中位 {med:7.1f}ms  周期均值 {avg:7.1f}ms  min {mn:7.1f}ms",
        flush=True,
    )

print()
stat = {}
for tag, key in (("OFF", False), ("ON ", True)):
    med = statistics.median([r[0] for r in res[key]])
    mn = min(r[2] for r in res[key])
    stat[key] = (med, mn)
    print(f"{tag}  中位 {med:7.1f}ms   min {mn:7.1f}ms")
for label, j in (("中位", 0), ("min ", 1)):
    b, a = stat[False][j], stat[True][j]
    print(f"{label} 口径 {b - a:+7.1f}ms  ({(b - a) / b * 100:+.1f}%)")
