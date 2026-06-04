#!/usr/bin/env python3
"""Parse logs/*.out and plot (1) val_loss trajectories and (2) bar of final val_loss per config."""
import glob, re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("img", exist_ok=True)
pat = re.compile(r"step:(\d+)/\d+ val_loss:([0-9.]+) train_time:(\d+)ms")

runs = {}
for f in sorted(glob.glob("logs/*.out")):
    name = os.path.basename(f)[:-4]
    pts = []
    with open(f) as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                pts.append((int(m.group(3)) / 1000.0, float(m.group(2))))
    if pts:
        runs[name] = pts

# trajectory plot
plt.figure(figsize=(9, 6))
for name, pts in sorted(runs.items()):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.plot(xs, ys, marker="o", ms=3, label=f"{name} ({ys[-1]:.4f})")
plt.xlabel("train time (s)")
plt.ylabel("val_loss")
plt.title("val_loss vs train time (5-min single-GPU budget)")
plt.ylim(3.5, 5.0)
plt.grid(alpha=0.3)
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("img/trajectories.png", dpi=110)

# final bar
finals = {n: p[-1][1] for n, p in runs.items()}
finals = dict(sorted(finals.items(), key=lambda kv: kv[1]))
plt.figure(figsize=(9, 5))
names = list(finals.keys())
vals = [finals[n] for n in names]
colors = ["#2a9d8f" if v == min(vals) else "#577590" for v in vals]
plt.barh(names, vals, color=colors)
for i, v in enumerate(vals):
    plt.text(v, i, f" {v:.4f}", va="center", fontsize=8)
plt.xlabel("final val_loss (lower = better)")
plt.xlim(3.5, max(vals) + 0.1)
plt.title("Final val_loss by config")
plt.tight_layout()
plt.savefig("img/final_bar.png", dpi=110)
print("wrote img/trajectories.png and img/final_bar.png")
print("finals:", {k: round(v, 4) for k, v in finals.items()})
