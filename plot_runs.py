# %%
"""
Plot loss curves from train_gpt2.py runs.

Each run writes a log to `logs/<run_id>.txt` containing lines like:
    step:5/5100 train_loss:4.1234 train_time:1234ms step_avg:12.34ms
    step:5/5100 val_loss:4.1234  train_time:1234ms step_avg:12.34ms

This notebook parses those lines and draws three stacked (vertical) plots for
every run in RUNS:
    1. loss vs step
    2. loss vs train tokens
    3. loss vs wall-clock time (train_time, the timed training seconds)

Set LIVE = True to re-read the logs and redraw on a loop (for a run in progress).
Run as a Jupyter-style notebook (the `# %%` cells) or as a plain script.
"""
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt

# ---- configure your runs here: {display name: path to the log .txt file} ----
RUNS = {
    "lr-0.001": "logs/12c5f72c-51ae-4b6a-b027-76521c0197c3.txt",
    "lr-0.003": "logs/ed2e2255-76b7-4ca4-97c2-13bec1c80c90.txt",
    "lr-0.004": "logs/996eb093-1854-4ff4-8186-9aba7a9d3fd3.txt",
    "lr-0.008": "logs/f7f517f4-0829-4413-a83e-8846ad991e6d.txt",
    # batch-size search (LR scaled as 0.004*sqrt(bs/512))
    "bs256_lr0.00283": "logs/008_bs256_lr0.00283_035e0a8c-5c8e-436a-8b1d-d2b8cac72d76.txt",
    "bs192_lr0.00245": "logs/014_bs192_lr0.0024495_0a84f15f-1a48-4220-8359-fad7b67b78db.txt",
    "bs128_lr0.002": "logs/011_bs128_lr0.002_82448c1c-1414-49a4-bbc7-adeb166fe44f.txt",
    "bs64_lr0.00141": "logs/012_bs64_lr0.0014142_b96290ae-4460-4de6-a579-70967576f852.txt",
    # LR search at bs128
    "bs128_lr0.00141": "logs/016_bs128_lr0.00141_6b3129b8-b3bd-4962-a97f-40b7d115072d.txt",
    "bs128_lr0.00283": "logs/015_bs128_lr0.00283_54c0f31c-6b13-4b96-a4e8-3ef8a011b569.txt",
    "bs128_lr0.004": "logs/017_bs128_lr0.004_740a973d-00af-40cc-bcae-06ab7f2a5e46.txt",
    "bs128_lr0.0048": "logs/021_bs128_lr0.0048_79bae3ae-7cdc-404e-93b6-4b8854e61bff.txt",
    "bs128_lr0.00566": "logs/018_bs128_lr0.00566_3f80324f-7092-4744-8d6d-7aa718a840dd.txt",
    "bs128_lr0.0058": "logs/023_bs128_lr0.0058_be4a334b-7df8-4cf3-891d-20d5751cb6a8.txt",
    "bs128_lr0.0062": "logs/022_bs128_lr0.0062_b67e436a-09c1-43a5-9a76-391fbcdbbb90.txt",
    "bs128_lr0.0067": "logs/020_bs128_lr0.0067_f1217391-9dac-4e53-af73-50ce8d38b2ef.txt",
    "bs128_lr0.008": "logs/019_bs128_lr0.008_4f90fb14-f6b8-4afa-9855-0334dc0657b1.txt",
    # LR search at bs64
    "bs64_lr0.002": "logs/024_bs64_lr0.002_ebd97d98-c93a-4cd5-87a6-3954bc560759.txt",
    "bs64_lr0.0025": "logs/028_bs64_lr0.0025_dc5ee5e5-0641-40d5-ad03-475b9fbbc6ea.txt",
    "bs64_lr0.003": "logs/025_bs64_lr0.003_112a1d35-822d-4056-aab8-c6b0993f581b.txt",
    "bs64_lr0.0035": "logs/029_bs64_lr0.0035_735f4b56-070a-44c4-bbe6-a549c36850b9.txt",
    "bs64_lr0.004": "logs/026_bs64_lr0.004_cec02a96-728d-498f-8d96-9619fc25de4a.txt",
    "bs64_lr0.006": "logs/027_bs64_lr0.006_5efcb70d-fe21-4475-9f9c-cafb3861018d.txt",
}

# refresh the plots every REFRESH_SECONDS while training is ongoing
LIVE = False
REFRESH_SECONDS = 5

# tokens per optimizer step = batch_size (sequences) * sequence_length (tokens).
# We try to read these from the code header embedded at the top of each log;
# these are the fallbacks if parsing fails.
DEFAULT_BATCH_SIZE = 8 * 64
DEFAULT_SEQUENCE_LENGTH = 1024


def _parse_hparam(code: str, name: str, default: int) -> int:
    """Pull `name : int = <expr>` out of the code header and eval the RHS."""
    m = re.search(rf"{name}\s*:\s*int\s*=\s*([0-9_*+\- ]+)", code)
    if not m:
        return default
    try:
        return int(eval(m.group(1), {"__builtins__": {}}))
    except Exception:
        return default


def parse_log(path: str):
    """Parse one log file into a dict of parallel lists.

    Returns keys:
        train_step, train_loss, train_time_s
        val_step,   val_loss,   val_time_s
        tokens_per_step
    train_time is converted from ms to seconds. Tokens for any point are
    step * tokens_per_step.
    """
    text = Path(path).read_text()
    tokens_per_step = (
        _parse_hparam(text, "batch_size", DEFAULT_BATCH_SIZE)
        * _parse_hparam(text, "sequence_length", DEFAULT_SEQUENCE_LENGTH)
    )

    out = {
        "train_step": [], "train_loss": [], "train_time_s": [],
        "val_step": [], "val_loss": [], "val_time_s": [],
        "tokens_per_step": tokens_per_step,
    }
    # e.g. "step:125/5100 val_loss:4.1234 train_time:8123ms step_avg:64.98ms"
    pat = re.compile(
        r"step:(\d+)/\d+\s+(train|val)_loss:([0-9.]+)\s+train_time:(\d+)ms"
    )
    for line in text.splitlines():
        m = pat.search(line)
        if not m:
            continue
        step = int(m.group(1))
        kind = m.group(2)
        loss = float(m.group(3))
        t_s = int(m.group(4)) / 1000.0
        out[f"{kind}_step"].append(step)
        out[f"{kind}_loss"].append(loss)
        out[f"{kind}_time_s"].append(t_s)
    return out


def draw(runs: dict, axes=None, ylim=(4, 7)):
    """Draw the four plots for all runs onto `axes` (created if None)."""
    if axes is None:
        _, axes = plt.subplots(4, 1, figsize=(8, 20))
    ax_step, ax_tok, ax_time, ax_bar = axes
    for ax in axes:
        ax.clear()

    # collected for the final bar chart: {name: (color, final_val_loss)}
    final_val = {}

    for name, path in runs.items():
        try:
            d = parse_log(path)
        except FileNotFoundError:
            print(f"[skip] {name}: log not found at {path}")
            continue

        tps = d["tokens_per_step"]
        # one color per run; val solid, train faint
        (line,) = ax_step.plot(
            d["val_step"], d["val_loss"], marker="o", label=f"{name} (val)"
        )
        color = line.get_color()
        ax_step.plot(
            d["train_step"], d["train_loss"], color=color, alpha=0.25,
            linewidth=1, label=f"{name} (train)",
        )

        ax_tok.plot(
            [s * tps for s in d["val_step"]], d["val_loss"],
            marker="o", color=color, label=f"{name} (val)",
        )
        ax_tok.plot(
            [s * tps for s in d["train_step"]], d["train_loss"],
            color=color, alpha=0.25, linewidth=1, label=f"{name} (train)",
        )

        ax_time.plot(
            d["val_time_s"], d["val_loss"], marker="o", color=color,
            label=f"{name} (val)",
        )
        ax_time.plot(
            d["train_time_s"], d["train_loss"], color=color, alpha=0.25,
            linewidth=1, label=f"{name} (train)",
        )

        if d["val_loss"]:
            final_val[name] = (color, d["val_loss"][-1])

    ax_step.set(xlabel="step", ylabel="loss", title="loss vs step", ylim=ylim)
    ax_tok.set(xlabel="train tokens", ylabel="loss", title="loss vs tokens", ylim=ylim)
    ax_time.set(
        xlabel="wall-clock train time (s)", ylabel="loss",
        title="loss vs wall-clock time",
        ylim=ylim
    )

    # fourth plot: bar chart of final validation loss per run
    names = list(final_val.keys())
    colors = [final_val[n][0] for n in names]
    losses = [final_val[n][1] for n in names]
    bars = ax_bar.bar(names, losses, color=colors)
    ax_bar.bar_label(bars, fmt="%.4f", fontsize=8)
    ax_bar.set(
        xlabel="run", ylabel="final val loss",
        title="final validation loss",
    )
    if losses:
        lo = min(losses)
        hi = max(losses)
        pad = max((hi - lo) * 0.15, 0.01)
        ax_bar.set_ylim(lo - pad, hi + pad)
    ax_bar.tick_params(axis="x", labelrotation=30)

    for ax in (ax_step, ax_tok, ax_time):
        ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return axes


if __name__ == "__main__":
    if LIVE:
        plt.ion()
        fig, axes = plt.subplots(4, 1, figsize=(8, 20))
        try:
            while True:
                draw(RUNS, axes)
                fig.canvas.draw_idle()
                plt.pause(REFRESH_SECONDS)
        except KeyboardInterrupt:
            print("stopped live refresh")
    else:
        draw(RUNS)
        plt.show()

# %%
# ---------------------------------------------------------------------------
# Final challenge: the two best configs (bs128 and bs64, each at its own tuned
# LR) run on a SINGLE GPU for the full 5-minute budget. The challenge must run
# on 1 GPU, so total_train_minutes=5.0 -> a full 300s wall-clock run each.
# Same four-plot style as above, but for just these two runs.
# ---------------------------------------------------------------------------
FINAL_RUNS = {
    "bs128 lr0.0058 (3.8969)":
        "logs/final_1gpu_bs128_lr0.0058_c5f1ca91-06b3-40ab-bbed-cc645f47d914.txt",
    "bs64 lr0.003 SDPA, no flex (3.8779)":
        "logs/final_1gpu_bs64_lr0.003_fd639e5c-9cf0-4cae-96d8-cf85d8927469.txt",
    # LR sweep at bs32 (single GPU, 5min)
    "bs32_lr0.0015": "logs/032_bs32_lr0.0015_6ad2b55c-70c5-4a35-93d7-91709f74cea5.txt",
    "bs32_lr0.0020": "logs/033_bs32_lr0.0020_8f51b28e-c26a-4c08-8810-0b2c7b223bf2.txt",
    "bs32_lr0.0028": "logs/031_bs32_lr0.0028_ca96468e-28b4-420f-aaa3-66a159032da4.txt",
    "bs32_lr0.0040": "logs/034_bs32_lr0.0040_94b1bfd0-056c-4bd8-86d2-5214fada0a02.txt",
    "bs32_lr0.0055": "logs/035_bs32_lr0.0055_4349ded9-3b71-48c1-b78d-9de790e00b07.txt",
    # LR sweep at bs16 (single GPU, 5min)
    "bs16_lr0.0010": "logs/036_bs16_lr0.0010_e5343848-f0fc-4947-9827-7370dca1e8c4.txt",
    "bs16_lr0.0015": "logs/037_bs16_lr0.0015_604e3aa6-8cb5-47c5-9583-b5697a25c504.txt",
    "bs16_lr0.0022": "logs/038_bs16_lr0.0022_bbfb6ac2-3905-486e-985a-58daed490ddb.txt",
    "bs16_lr0.0030": "logs/039_bs16_lr0.0030_185bd7ac-d6ef-495d-8033-db8b22553cd7.txt",
    # --- strong bs64 (standard arch, the optimal config) ---
    "bs64 lr0.0035 (3.8831)": "logs/045_bs64_lr0.0035_c24ff600-d463-4fc3-9f33-6309d49fdf4a.txt",
    "bs64 lr0.0045 (3.8831)": "logs/046_bs64_lr0.0045_ca19be1a-6350-4757-8886-39b1f010eb37.txt",
    # --- larger batch: bs96 ---
    "bs96 lr0.0040 (3.8855)": "logs/047_bs96_lr0.0040_2886f8a6-792e-4220-8a56-a310d168101c.txt",
    "bs96 lr0.0050 (3.8920)": "logs/048_bs96_lr0.0050_6b4c14fe-4a63-4e31-bff7-a835b36f9489.txt",
    # --- doc-mask, bs64: flex attention vs none (and compiled flex) ---
    "doc-mask, no flex (3.9088)": "logs/042_docmask-base-1gpu-bs64_a6af56ee-1408-4f54-a2ba-abe540b189b8.txt",
    "doc-mask + flex (3.9219)": "logs/043_docmask-flex-bs64_7a9624ce-ed1c-4bda-b4fe-1e1956fbd8a4.txt",
    "doc-mask + flex, compiled (3.8871)": "logs/054_docmask-flex-compiled-bs64_d7cc9d56-3de4-4cfc-ac58-c49408850378.txt",
    # train_gpt2.py (compiled flex + doc-mask + optimized data loader) at the best bs64/lr0.003
    "flex compiled, bs64 lr0.003 (BEST 3.8667)": "logs/066_flex_compiled_bs64_lr0.003_dbs64_1c3e64d8-4e48-4a71-ab92-c0a992da0c86.txt",
    # --- cut-cross-entropy (fused linear+CE) vs standard CE (= bs64 lr0.0035 above) ---
    "CCE fused linear+CE (3.9791)": "logs/061_cce_bs64_lr0.0035_dbs64_6d281ffd-1503-459a-ab9e-6528e6452ce9.txt",
    # --- fp8 lm-head vs bf16 (= bs64 lr0.0035 above) ---
    "fp8 lm-head (3.8973)": "logs/065_fp8head_bs64_dbs64_rerun_0fb07ba6-043b-4722-b901-1ab6c6a3d609.txt",
}

# these full-length runs converge well below the 75s runs, so use a tighter window
_, final_axes = plt.subplots(4, 1, figsize=(8, 20))
draw(FINAL_RUNS, final_axes, ylim=(3.7, 6.0))
plt.show()

# %%
