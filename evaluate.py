import argparse
import csv
import os

import torch
import yaml

from models.registry import get_model, list_architectures
from data.loaders import get_loaders, get_corruption_loader, check_cifar_c, CORRUPTIONS, CORRUPTION_GROUPS
from evaluation.metrics import (collect, accuracy, top5_accuracy, ece,
                                 compute_baseline_errors, corruption_error,
                                 measure_latency, model_size_mb, compression_ratio, param_count)


def ckpt_path(cfg, dataset, arch, config_id):
    for ext in (".pth", ".pt"):
        p = os.path.join(cfg["checkpoints_dir"], dataset, arch, f"{config_id}{ext}")
        if os.path.exists(p):
            return p, ext == ".pt"
    return None, False


def load_model(cfg, dataset, arch, config_id):
    path, is_q = ckpt_path(cfg, dataset, arch, config_id)
    if path is None:
        return None, False
    if is_q:
        model = torch.jit.load(path, map_location="cpu")
    else:
        model = get_model(arch, num_classes=10 if dataset == "cifar10" else 100)
        model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval(), is_q


def all_configs(cfg):
    configs = [{"id": "baseline", "label": "FP32 Baseline", "method": "none"}]
    for key in ("phase1_configs", "phase2_configs", "phase3_configs"):
        configs += [c for c in cfg.get(key, []) if c["id"] != "baseline"]
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--arch",    default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["results_dir"], exist_ok=True)
    archs  = [args.arch] if args.arch else list_architectures()
    ev_cfg = cfg["evaluation"]
    device = "cpu"
    rows   = []

    for arch in archs:
        print(f"\n{'='*60}  {arch}  {'='*60}")

        baseline_model, _ = load_model(cfg, args.dataset, arch, "baseline")
        if baseline_model is None:
            print(f"  [SKIP] No baseline for {arch}.")
            continue

        baseline_path, _ = ckpt_path(cfg, args.dataset, arch, "baseline")
        baseline_size    = model_size_mb(baseline_path)
        baseline_errors  = {}

        if check_cifar_c(cfg["data_dir"], args.dataset):
            def bl_fn(c, s):
                return get_corruption_loader(args.dataset, c, s, cfg["data_dir"],
                                              batch_size=ev_cfg["batch_size"],
                                              num_workers=cfg["num_workers"])
            print("  Computing baseline corruption errors ...")
            baseline_errors = compute_baseline_errors(
                baseline_model, bl_fn, CORRUPTIONS,
                severities=ev_cfg["corruption_severities"], device=device
            )

        for comp_cfg in all_configs(cfg):
            cid         = comp_cfg["id"]
            label       = comp_cfg["label"]
            model, is_q = load_model(cfg, args.dataset, arch, cid)
            if model is None:
                print(f"  [SKIP] {label}")
                continue

            print(f"\n  -- {label} --")
            path, _  = ckpt_path(cfg, args.dataset, arch, cid)
            size     = model_size_mb(path)
            cr       = compression_ratio(baseline_size, size)
            latency  = measure_latency(model, n_runs=ev_cfg["latency_runs"], device=device)
            n_params = param_count(model) if not is_q else float("nan")

            _, test_loader = get_loaders(args.dataset, cfg["data_dir"],
                                          batch_size=ev_cfg["batch_size"],
                                          num_workers=cfg["num_workers"])
            probs_clean, labels_clean = collect(model, test_loader, device=device)
            acc_clean  = accuracy(probs_clean, labels_clean)
            top5_clean = top5_accuracy(probs_clean, labels_clean)
            ece_clean  = ece(probs_clean, labels_clean, ev_cfg["ece_bins"])

            row = {
                "dataset":           args.dataset,
                "arch":              arch,
                "config_id":         cid,
                "config_label":      label,
                "is_quantized":      is_q,
                "size_mb":           round(size, 3),
                "compression_ratio": round(cr, 3),
                "n_params":          n_params,
                "latency_ms":        round(latency, 3),
                "acc_clean":         round(acc_clean, 3),
                "top5_clean":        round(top5_clean, 3),
                "ece_clean":         round(ece_clean, 5),
            }

            if check_cifar_c(cfg["data_dir"], args.dataset):
                def loader_fn(c, s):
                    return get_corruption_loader(args.dataset, c, s, cfg["data_dir"],
                                                  batch_size=ev_cfg["batch_size"],
                                                  num_workers=cfg["num_workers"])
                ce_per, mce = corruption_error(model, loader_fn, baseline_errors, CORRUPTIONS,
                                                severities=ev_cfg["corruption_severities"], device=device)
                row["mce"] = round(mce, 3)
                for group, members in CORRUPTION_GROUPS.items():
                    row[f"mce_{group}"] = round(100.0 * sum(ce_per[c] for c in members) / len(members), 3)
                for c in CORRUPTIONS:
                    for s in ev_cfg["corruption_severities"]:
                        probs_c, lbls_c    = collect(model, loader_fn(c, s), device=device)
                        acc_c              = accuracy(probs_c, lbls_c)
                        row[f"acc_{c}_s{s}"]  = round(acc_c, 3)
                        row[f"drop_{c}_s{s}"] = round(acc_clean - acc_c, 3)
                        row[f"ece_{c}_s{s}"]  = round(ece(probs_c, lbls_c, ev_cfg["ece_bins"]), 5)
            else:
                row["mce"] = float("nan")

            rows.append(row)
            print(f"    acc_clean: {acc_clean:.2f}%  mCE: {row['mce']}  latency: {latency:.2f}ms  size: {size:.2f}MB")

    out_csv = os.path.join(cfg["results_dir"], f"{args.dataset}_results.csv")
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved -> {out_csv}")


if __name__ == "__main__":
    main()
