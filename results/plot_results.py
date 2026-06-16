import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]
MARKERS = {"resnet20":"o","resnet50":"s","vgg19":"^","mobilenetv2":"D","efficientnet_b0":"P","convnext_tiny":"*"}


def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.basename(path)}")


def load_df(results_dir, dataset):
    path = os.path.join(results_dir, f"{dataset}_results.csv")
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found.")
        sys.exit(1)
    return pd.read_csv(path)


def grouped_bar(df, col, ylabel, title, fname, out_dir, ref_line=None):
    archs   = df["arch"].unique()
    configs = df["config_id"].unique()
    x       = np.arange(len(archs))
    w       = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, cid in enumerate(configs):
        sub   = df[df["config_id"] == cid].set_index("arch").reindex(archs)
        label = df[df["config_id"] == cid]["config_label"].iloc[0]
        ax.bar(x + i * w - 0.4 + w / 2, sub[col], w, label=label, color=PALETTE[i % len(PALETTE)])
    if ref_line is not None:
        ax.axhline(ref_line, ls="--", color="black", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(archs, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    savefig(fig, os.path.join(out_dir, fname))


def plot_drop_heatmap(df, out_dir):
    from data.loaders import CORRUPTIONS
    drop_cols = [c for c in df.columns if c.startswith("drop_") and "_s" in c]
    if not drop_cols:
        return
    for arch in df["arch"].unique():
        sub   = df[df["arch"] == arch]
        pivot = sub.set_index("config_label")[drop_cols].astype(float)
        fig, ax = plt.subplots(figsize=(max(12, len(drop_cols) * 0.4), max(5, len(pivot) * 0.5)))
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=15)
        plt.colorbar(im, ax=ax, label="Accuracy Drop (pp)")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([c.replace("drop_","").replace("_s","\ns=") for c in pivot.columns], rotation=90, fontsize=6)
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f"Accuracy Drop: {arch}")
        savefig(fig, os.path.join(out_dir, f"acc_drop_heatmap_{arch}.png"))


def plot_ece(df, out_dir):
    ece_cols = [c for c in df.columns if c.startswith("ece_") and "_s" in c]
    if not ece_cols:
        return
    df2   = df.copy()
    df2["ece_mean_corrupted"] = df2[ece_cols].mean(axis=1)
    archs = df2["arch"].unique()
    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 5))
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        sub = df2[df2["arch"] == arch]
        x   = np.arange(len(sub))
        ax.bar(x - 0.2, sub["ece_clean"],          0.4, label="Clean",     color="#4C72B0")
        ax.bar(x + 0.2, sub["ece_mean_corrupted"], 0.4, label="Corrupted", color="#DD8452")
        ax.set_xticks(x); ax.set_xticklabels(sub["config_label"], rotation=30, ha="right", fontsize=7)
        ax.set_title(arch); ax.set_ylabel("ECE"); ax.legend(fontsize=8)
    fig.suptitle("ECE: Clean vs. Mean Corrupted")
    savefig(fig, os.path.join(out_dir, "ece_clean_vs_corrupted.png"))


def plot_latency_size(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    norm    = plt.Normalize(df["acc_clean"].min(), df["acc_clean"].max())
    for arch in df["arch"].unique():
        sub = df[df["arch"] == arch]
        sc  = ax.scatter(sub["size_mb"], sub["latency_ms"],
                         c=sub["acc_clean"], cmap="viridis", norm=norm,
                         s=80, marker=MARKERS.get(arch, "o"),
                         label=arch, edgecolors="black", linewidths=0.5, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="Clean Accuracy (%)")
    ax.set_xlabel("Model Size (MB)"); ax.set_ylabel("Latency (ms/sample)")
    ax.set_title("Efficiency Trade-off: Latency vs. Model Size")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(out_dir, "latency_vs_size_scatter.png"))


def plot_pareto(df, out_dir):
    if "mce" not in df.columns or "compression_ratio" not in df.columns:
        return
    for arch in df["arch"].unique():
        sub = df[df["arch"] == arch].dropna(subset=["mce","compression_ratio"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        norm    = plt.Normalize(sub["acc_clean"].min(), sub["acc_clean"].max())
        sc      = ax.scatter(sub["compression_ratio"], sub["mce"],
                             c=sub["acc_clean"], cmap="viridis", norm=norm,
                             s=100, edgecolors="black", linewidths=0.5)
        plt.colorbar(sc, ax=ax, label="Clean Accuracy (%)")
        for _, row in sub.iterrows():
            ax.annotate(row["config_label"], (row["compression_ratio"], row["mce"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
        pts       = sub[["compression_ratio","mce"]].values
        dominated = np.zeros(len(pts), dtype=bool)
        for i in range(len(pts)):
            for j in range(len(pts)):
                if i != j and pts[j,0] >= pts[i,0] and pts[j,1] <= pts[i,1]:
                    dominated[i] = True; break
        pareto = pts[~dominated]; pareto = pareto[pareto[:,0].argsort()]
        ax.plot(pareto[:,0], pareto[:,1], "r--", lw=1.5, label="Pareto front")
        ax.axhline(100, ls=":", color="gray", alpha=0.5, label="Baseline mCE")
        ax.set_xlabel("Compression Ratio"); ax.set_ylabel("mCE (lower = better)")
        ax.set_title(f"Pareto Front: {arch}"); ax.legend(fontsize=8)
        savefig(fig, os.path.join(out_dir, f"pareto_front_{arch}.png"))


def plot_sweep(df, out_dir, config_ids, x_vals, xlabel, fname):
    sub = df[df["config_id"].isin(config_ids)]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for arch in sub["arch"].unique():
        s = sub[sub["arch"] == arch].set_index("config_id").reindex(config_ids)
        axes[0].plot(x_vals, s["acc_clean"], marker="o", label=arch)
        axes[1].plot(x_vals, s["mce"],       marker="o", label=arch)
    for ax, ylabel, title in zip(axes,
                                  ["Clean Accuracy (%)","mCE"],
                                  [f"Accuracy vs. {xlabel}", f"Robustness vs. {xlabel}"]):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8)
    savefig(fig, os.path.join(out_dir, fname))


def plot_radar(df, out_dir):
    groups = [g for g in ["mce_noise","mce_blur","mce_weather","mce_digital"] if g in df.columns]
    if not groups:
        return
    labels = [g.replace("mce_","").capitalize() for g in groups]
    n      = len(groups)
    angles = [k / n * 2 * np.pi for k in range(n)] + [0]
    for arch in df["arch"].unique():
        sub = df[df["arch"] == arch].dropna(subset=groups)
        fig = plt.figure(figsize=(7, 7))
        ax  = fig.add_subplot(111, polar=True)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
        for i, (_, row) in enumerate(sub.iterrows()):
            vals = [row[g] for g in groups] + [row[groups[0]]]
            ax.plot(angles, vals, color=PALETTE[i % len(PALETTE)], lw=1.5, label=row["config_label"])
            ax.fill(angles, vals, color=PALETTE[i % len(PALETTE)], alpha=0.1)
        ax.axhline(100, ls="--", color="gray", alpha=0.4)
        ax.set_title(f"Corruption Group mCE: {arch}", pad=20)
        ax.legend(bbox_to_anchor=(1.3, 1.0), fontsize=7)
        savefig(fig, os.path.join(out_dir, f"corruption_group_radar_{arch}.png"))


def plot_severity_curves(df, out_dir):
    from data.loaders import CORRUPTIONS
    severities = [1, 2, 3, 4, 5]
    for arch in df["arch"].unique():
        sub  = df[df["arch"] == arch]
        n    = len(CORRUPTIONS)
        cols = 5
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.array(axes).flatten()
        for ax, corr in zip(axes, CORRUPTIONS):
            drop_cols = [f"drop_{corr}_s{s}" for s in severities]
            if not all(c in sub.columns for c in drop_cols):
                continue
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.plot(severities, [row[c] for c in drop_cols], marker="o",
                        label=row["config_label"], color=PALETTE[i % len(PALETTE)], lw=1.5)
            ax.set_title(corr.replace("_"," ").title(), fontsize=9)
            ax.set_xlabel("Severity"); ax.set_ylabel("Acc Drop (pp)"); ax.grid(True, alpha=0.3)
        for ax in axes[n:]:
            ax.set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Accuracy Drop vs. Severity: {arch}", fontsize=13)
        plt.tight_layout()
        savefig(fig, os.path.join(out_dir, f"severity_curves_{arch}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10","cifar100"])
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    df = load_df(out_dir, args.dataset)
    print(f"Loaded {len(df)} rows\n")
    grouped_bar(df, "acc_clean", "Clean Accuracy (%)", "Clean Test Accuracy",      "clean_accuracy_by_arch.png", out_dir)
    grouped_bar(df, "mce",       "mCE",                "Mean Corruption Error",    "mce_by_arch.png",            out_dir, ref_line=100)
    plot_drop_heatmap(df, out_dir)
    plot_ece(df, out_dir)
    plot_latency_size(df, out_dir)
    plot_pareto(df, out_dir)
    plot_sweep(df, out_dir, ["pruning_20","pruning_50","pruning_70","pruning_90"], [20,50,70,90], "Pruning Sparsity (%)", "sparsity_sweep.png")
    plot_sweep(df, out_dir, ["kd_temp2","kd_temp4","kd_temp8"],                   [2,4,8],        "KD Temperature",       "kd_temperature_sweep.png")
    plot_radar(df, out_dir)
    plot_severity_curves(df, out_dir)
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
