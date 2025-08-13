import argparse
from pathlib import Path
import math
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv
from torch.profiler import profile, ProfilerActivity

from beat_this.dataset.dataset import BeatDataModule
from beat_this.inference import streaming_predict, load_model

def export_step_times_csv(path, step_times):
    # keep only real step rows; drop "mark" rows
    rows = [r for r in step_times if "mark" not in r]
    if not rows:
        print("No step timings to export.")
        return
    # normalize keys and handle missing gpu_ms
    for r in rows:
        r.setdefault("gpu_ms", "") # empty if cpu
        r.setdefault("late_ms", 0.0)
        r.setdefault("late_flag", 0)
        r.setdefault("budget_ms", 0.0)
        r.setdefault("track_idx", 0)
        r.setdefault("step_abs", 0)
    fieldnames = ["step_abs","track_idx","new_frames","wall_ms","gpu_ms","budget_ms","late_ms","late_flag"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Wrote {len(rows)} step rows to {path}")

def plot_latency(step_times, hop_ms, out_path="latency_plot.png", warmup=0):
    """Line plot: per-step budget vs actual wall time. shaded lateness where actual > budget."""
    if not step_times:
        print("No step timings to plot.")
        return
    steps = step_times[warmup:] if warmup else step_times

    budgets = [st["new_frames"] * hop_ms for st in steps] # in ms
    walls = [st["wall_ms"] for st in steps] # in ms
    x = np.arange(1, len(steps) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(x, budgets, label=f"Budget per step ({hop_ms:.1f} ms/frame)")
    plt.plot(x, walls, label="Actual step time")

    # shade lateness
    over = np.array(walls) > np.array(budgets)
    if over.any():
        plt.fill_between(x, budgets, walls, where=over, alpha=0.3, interpolate=True, label="Lateness")

    plt.xlabel("Step")
    plt.ylabel("Milliseconds")
    plt.ylim([-1, 100])
    plt.title("Streaming latency vs. per-step budget")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved latency plot to {out_path}")

def percentile_nearest_rank(sorted_vals, p):
    if not sorted_vals: return float("nan")
    n = len(sorted_vals)
    idx = max(0, min(n-1, math.ceil(p * n) - 1))
    return sorted_vals[idx]

def summarize(step_times, hop_ms, eps=0.05):
    # period can vary if new_frames varies (e.g. last step)
    periods = [(st["new_frames"] * hop_ms) for st in step_times]
    walls = [st["wall_ms"] for st in step_times]
    lates = [st["late_ms"] for st in step_times]
    n = len(step_times)

    # on-time vs late
    late_mask = [lt > eps for lt in lates]
    late_vals = [lt for lt in lates if lt > eps]
    late_count = sum(late_mask)
    late_frac = late_count / n if n else float("nan")
    late_sum = sum(late_vals)

    # late streak (burstiness)
    max_streak = streak = 0
    for is_late in late_mask:
        streak = streak + 1 if is_late else 0
        if streak > max_streak: max_streak = streak

    # percentiles
    walls_sorted = sorted(walls)
    p50 = percentile_nearest_rank(walls_sorted, 0.50)
    p95 = percentile_nearest_rank(walls_sorted, 0.95)
    p99 = percentile_nearest_rank(walls_sorted, 0.99)
    wmax = walls_sorted[-1] if walls_sorted else float("nan")

    late_p50 = percentile_nearest_rank(sorted(late_vals), 0.50)
    late_p95 = percentile_nearest_rank(sorted(late_vals), 0.95)
    late_p99 = percentile_nearest_rank(sorted(late_vals), 0.99)
    late_max = max(late_vals) if late_vals else 0.0

    # utilization (use the median period so mixed new_frames doesn’t skew)
    period_med = percentile_nearest_rank(sorted(periods), 0.50)
    util_p50 = (p50 / period_med) if period_med else float("nan")

    return {
        "steps": n,
        "on_time_frac": 1 - late_frac,
        "late_count": late_count,
        "late_frac": late_frac,
        "late_ms_sum": late_sum,
        "late_ms_p50": late_p50,
        "late_ms_p95": late_p95,
        "late_ms_p99": late_p99,
        "late_ms_max": late_max,
        "wall_ms_p50": p50,
        "wall_ms_p95": p95,
        "wall_ms_p99": p99,
        "wall_ms_max": wmax,
        "util_p50": util_p50,
        "late_max_streak": max_streak,
    }

def main(args):
    assert args.device in ["cpu", "cuda"]
    print(args)
    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    ckpt_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "checkpoints"
    ckpt_path = ckpt_dir / args.model_ckpt

    model, hparams = load_model(ckpt_path, device=args.device, return_hparams=True)
    hop_ms = 1000/hparams["fps"]

    dm = BeatDataModule(
        data_dir=data_dir,
        spect_fps=hparams["fps"],
        test_dataset="gtzan",
        fold=None,
        predict_datasplit="val", # we're intending to predict on the full songs in the validation set
        num_workers=8
    )
    dm.setup("predict")
    pred_loader = dm.predict_dataloader()

    step_times = []
    track_idx = 0
    def on_step(**kw):
        budget_ms = kw["new_frames"] * hop_ms
        late_flag = 1 if kw["late_ms"] > 0.05 else 0  # 0.05 ms tolerance
        kw.update({
            "budget_ms": budget_ms,
            "late_flag": late_flag,
            "track_idx": track_idx,
            "step_abs": len(step_times),  # global step index across all tracks
        })
        step_times.append(kw)

    #with profile(
    #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #    record_shapes=True,  # show input shapes per op
    #    profile_memory=True,  # track allocs
    #    with_stack=False  # True = Python call stacks (slower)
    #) as prof:
    limit = 20
    for batch in tqdm(pred_loader, total=len(pred_loader), desc="Streaming (val)", unit="track"):
        track_id = f"track_{track_idx}"
        spect = batch["spect"]  # (1, T, F)
        out = streaming_predict(
            model,
            spect=spect,
            window_size=args.window_size,
            peek_size=args.peek_size,
            device=args.device,
            tolerance=3,
            report=args.report,
            hop_ms=hop_ms,
            pace=args.pace,
            on_step=on_step,
        )
        track_idx += 1
        if limit and track_idx >= limit:
            break

    '''print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=30
    ))'''

    if step_times and args.report:
        s = summarize(step_times, hop_ms)
        print(f"Model: {ckpt_path.name}  Device: {args.device}  Window size: {args.window_size}  Peek size: {args.peek_size}")
        print(
            f"steps={s['steps']} on_time={s['on_time_frac'] * 100:.1f}%  "
            f"wall-ms p50/p95/p99={s['wall_ms_p50']:.2f}/{s['wall_ms_p95']:.2f}/{s['wall_ms_p99']:.2f}  "
            f"late frac={s['late_frac'] * 100:.2f}% p95={s['late_ms_p95']:.2f}ms max={s['late_ms_max']:.2f}ms  "
            f"util p50={s['util_p50'] * 100:.1f}%  burst(max_streak)={s['late_max_streak']}"
        )
        plot_latency(step_times, hop_ms, out_path="latency_plot.png", warmup=30)
        if args.export_csv:
            export_step_times_csv(args.export_csv, step_times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Emulates real-time audio streaming for the purpose of latency testing. Expects precomputed spectrograms"
                    "as we're only interested in the latency of the model inference, not the audio processing."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run the model on (default: gpu). If set to 'gpu', it will use the first available GPU, otherwise it will use CPU."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Size of the audio window (in frames) to process at a time (default: 512)."
    )
    parser.add_argument(
        "--peek-size",
        type=int,
        default=2,
        help="Number of frames to peek ahead in the audio stream, i.e., nr of new frames per iteration (default: 2)."
    )
    parser.add_argument(
        "--cache-conv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to cache convolutional layers for faster inference (default: False)."
    )
    parser.add_argument(
        "--cache-kv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to cache keys and values for attention for faster inference (default: False)."
    )
    parser.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
        help="Path to the model checkpoint file."
    )
    parser.add_argument(
        "--pace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to emulate the real-time pace of an audio stream by sleeping after processing a chunk (default: False)."
    )
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to report latency statistics (default: False)."
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="If set, write per step timings to csv."
    )

    args = parser.parse_args()

    main(args)