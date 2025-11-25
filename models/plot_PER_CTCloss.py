import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

MODEL_CHECKPOINT_DIR = "trained_models/baseline_gru"
LOG_PATH = f"{MODEL_CHECKPOINT_DIR}/training_log"
INTERVAL = 2000            # show train_loss averaged every N global batches
FS_ANN = 8                 # annotation fontsize

# --- regexes aligned to your log ---
RE_TRAIN = re.compile(r"Train batch\s*(\d+)\s*:?\s*loss:\s*([\d\.]+)")
RE_VAL   = re.compile(r"Val batch\s*(\d+)\s*:\s*PER \(avg\):\s*([\d\.]+)\s*CTC Loss \(avg\):\s*([\d\.]+)")

def parse_log(path: str):
    train, val = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = RE_TRAIN.search(line)
            if m:
                train.append(("train", int(m.group(1)), float(m.group(2))))
                continue
            m = RE_VAL.search(line)
            if m:
                val.append(("val", int(m.group(1)), float(m.group(2)), float(m.group(3))))
    df_tr = pd.DataFrame(train, columns=["kind", "batch", "loss"])
    df_v  = pd.DataFrame(val,   columns=["kind", "batch", "per", "loss"])
    return df_tr, df_v

def split_runs_by_resets(batches):
    """Split the raw batch sequence into runs; a new run starts when the id decreases."""
    runs = []
    if len(batches) == 0:
        return runs
    start = 0
    last  = batches[0]
    for i in range(1, len(batches)):
        if batches[i] < last:       # reset -> new run
            runs.append((start, i-1))
            start = i
        last = batches[i]
    runs.append((start, len(batches)-1))
    return runs

def make_global_batches(df):
    """
    Create a strictly increasing 'g_batch' by:
      - splitting into runs (counter resets),
      - computing an offset for each run: sum(prev_run_max + 1),
      - global = raw_batch + offset_of_run.
    """
    if df.empty:
        df["g_batch"] = []
        return df

    batches = df["batch"].tolist()
    runs = split_runs_by_resets(batches)

    # compute per-run max to build offsets
    run_max = [max(batches[s:e+1]) for (s, e) in runs]
    offsets = [0]
    for i in range(1, len(runs)):
        offsets.append(offsets[i-1] + run_max[i-1] + 1)

    g = [0] * len(batches)
    for run_idx, (s, e) in enumerate(runs):
        off = offsets[run_idx]
        for i in range(s, e+1):
            g[i] = batches[i] + off
    df = df.copy()
    df["g_batch"] = g
    return df

def average_train_right_closed(df_train, interval):
    """
    Average train loss per interval using RIGHT-CLOSED bins labeled by the UPPER edge.
    Bins: (-1, 0], (0, interval], (interval, 2*interval], ...
    The last bin’s right edge is floored to the nearest multiple so the final label is 12000.
    """
    if df_train.empty:
        return pd.DataFrame(columns=["g_batch", "loss"])

    # max_edge = int((df_train["g_batch"].max() // interval) * interval)  # floor to keep 12000, not 14000
    # Build edges so that 0 falls into the first bin (-1, 0], then right-closed every interval.
    max_edge = int((df_train["g_batch"].max() + interval - 1) // interval * interval)
    edges = [-1, 0] + list(range(interval, max_edge + 1, interval))     # e.g., [-1,0,2000,4000,...,12000]
    labels = edges[1:]                                                  # right-edge labels: 0,2000,...,12000

    cats = pd.cut(
        df_train["g_batch"],
        bins=edges,
        right=True,
        include_lowest=True,
        labels=labels
    )
    out = (df_train.groupby(cats, observed=True)["loss"]
                   .mean()
                   .reset_index())
    out.columns = ["g_batch", "loss"]
    out["g_batch"] = out["g_batch"].astype(int)
    return out

def main():
    df_train, df_val = parse_log(LOG_PATH)
    # Build global batch indices independently for train and val (based on their own resets)
    df_train = make_global_batches(df_train)
    df_val   = make_global_batches(df_val)

    # --- average train loss using right-closed bins (ensures a point at 12000) ---
    df_train_avg = average_train_right_closed(df_train, INTERVAL)

    # ======================= FIGURE 1: Loss =======================
    plt.figure(figsize=(12, 6))
    if not df_train_avg.empty:
        plt.plot(df_train_avg["g_batch"], df_train_avg["loss"],
                 "go-", label=f"Train Loss (avg / {INTERVAL} batches)")
        for x, y in zip(df_train_avg["g_batch"], df_train_avg["loss"]):
            plt.annotate(f"{y:.2f}", (x, y), xytext=(0, 6),
                         textcoords="offset points", ha="center",
                         fontsize=FS_ANN, color="green")

    if not df_val.empty:
        plt.plot(df_val["g_batch"], df_val["loss"], "r.-", linewidth=1, markersize=4,
                 label="Val CTC Loss")
        for x, y in zip(df_val["g_batch"], df_val["loss"]):
            plt.annotate(f"{y:.2f}", (x, y), xytext=(0, 6),
                         textcoords="offset points", ha="center",
                         fontsize=FS_ANN, color="red")

    plt.xlabel("Global Training Batch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation CTC Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{MODEL_CHECKPOINT_DIR}/train_v_val_ctc_loss')

    # ======================= FIGURE 2: PER =======================
    if not df_val.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(df_val["g_batch"], df_val["per"], "b.-", linewidth=1, markersize=4,
                 label="PER (avg)")
        for x, y in zip(df_val["g_batch"], df_val["per"]):
            plt.annotate(f"{y:.3f}", (x, y), xytext=(0, 6),
                         textcoords="offset points", ha="center",
                         fontsize=FS_ANN, color="blue")
        plt.xlabel("Global Training Batch")
        plt.ylabel("PER")
        plt.title("Validation PER")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{MODEL_CHECKPOINT_DIR}/val_per_plot')

if __name__ == "__main__":
    main()
