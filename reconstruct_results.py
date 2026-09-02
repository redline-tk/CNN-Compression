import os
import json
import csv
import yaml

WANDB_DIR = "wandb"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
SEVERITIES = [1, 2, 3, 4, 5]
GROUPS = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

rows_by_dataset = {}

for run_dir in sorted(os.listdir(WANDB_DIR)):
    if not run_dir.startswith("run-"):
        continue
    summary_path = os.path.join(WANDB_DIR, run_dir, "files", "wandb-summary.json")
    config_path  = os.path.join(WANDB_DIR, run_dir, "files", "config.yaml")
    if not os.path.exists(summary_path) or not os.path.exists(config_path):
        continue
    with open(summary_path) as f:
        summary = json.load(f)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    arch      = (config.get("arch")      or {}).get("value")
    dataset   = (config.get("dataset")   or {}).get("value")
    config_id = (config.get("config_id") or {}).get("value")
    if not arch or not dataset or not config_id:
        continue
    if not summary.get("acc_clean"):
        continue
    row = {
        "dataset":           dataset,
        "arch":              arch,
        "config_id":         config_id,
        "config_label":      config_id,
        "is_quantized":      summary.get("is_quantized", False),
        "size_mb":           summary.get("size_mb", float("nan")),
        "compression_ratio": summary.get("compression_ratio", float("nan")),
        "n_params":          summary.get("n_params", float("nan")),
        "latency_ms":        summary.get("latency_ms", float("nan")),
        "acc_clean":         summary.get("acc_clean", float("nan")),
        "top5_clean":        summary.get("top5_clean", float("nan")),
        "ece_clean":         summary.get("ece_clean", float("nan")),
        "mce":               summary.get("mce", float("nan")),
    }
    for g in GROUPS:
        row["mce_" + g] = summary.get("mce_" + g, float("nan"))
    for c in CORRUPTIONS:
        for s in SEVERITIES:
            row["acc_"  + c + "_s" + str(s)] = summary.get("acc_"  + c + "_s" + str(s), float("nan"))
            row["drop_" + c + "_s" + str(s)] = summary.get("drop_" + c + "_s" + str(s), float("nan"))
            row["ece_"  + c + "_s" + str(s)] = summary.get("ece_"  + c + "_s" + str(s), float("nan"))

    key = dataset + "_" + arch + "_" + config_id
    existing = rows_by_dataset.get(dataset, {}).get(key)
    if existing is None or (float("nan") == existing.get("mce", float("nan")) and row.get("mce") == row.get("mce")):
        rows_by_dataset.setdefault(dataset, {})[key] = row

for dataset, rows_dict in rows_by_dataset.items():
    rows = list(rows_dict.values())
    if not rows:
        continue
    out_path = os.path.join(OUTPUT_DIR, dataset + "_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote " + str(len(rows)) + " rows -> " + out_path)

print("Done.")
