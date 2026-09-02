import pandas as pd
import numpy as np
import os

ARCH_LABELS = {
    "resnet20":        "ResNet-20",
    "resnet50":        "ResNet-50",
    "vgg19":           "VGG-19",
    "mobilenetv2":     "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny":   "ConvNeXt-Tiny",
}

METHOD_LABELS = {
    "baseline":    "FP32 Baseline",
    "pruning":     "Pruning",
    "pruning_kd":  "Pruning + KD",
    "pruning_qat": "Pruning + QAT",
    "ptq":         "PTQ",
    "qat":         "QAT",
    "quant_int8":                    "Quant INT8",
    "sparsity":                      "Sparsity",
    "sparsity_quant_int8":           "Sparsity + Quant INT8",
    "clustering":                    "Clustering",
    "clustering_quant_int8":         "Clustering + Quant INT8",
    "sparsity_clustering":           "Sparsity-preserving Clustering",
    "sparsity_clustering_quant_int8":"SPC + INT8",
    "qat_only":    "QAT",
    "qat_int8":    "QAT + INT8",
    "cqat":        "CQAT",
    "cqat_int8":   "CQAT + INT8",
    "pqat":        "PQAT",
    "pqat_int8":   "PQAT + INT8",
    "pcqat":       "PCQAT",
    "pcqat_int8":  "PCQAT + INT8",
    "pruning_20":  r"Pruning 20\%",
    "pruning_50":  r"Pruning 50\%",
    "pruning_70":  r"Pruning 70\%",
    "pruning_90":  r"Pruning 90\%",
    "kd_temp2":    r"KD $T=2$",
    "kd_temp4":    r"KD $T=4$",
    "kd_temp8":    r"KD $T=8$",
    "structured_pruning_30": r"Structured Pruning 30\%",
    "structured_pruning_50": r"Structured Pruning 50\%",
    "pruning_kd_70": r"Pruning 70\% + KD",
    "fp16":        "FP16",
}


def fmt(val, decimals=2):
    if pd.isna(val):
        return r"\textemdash"
    return f"{val:.{decimals}f}"


def make_phase1_table(csv_path, label, caption):
    df = pd.read_csv(csv_path)
    configs = ["baseline","pruning","ptq","qat","pruning_qat","pruning_kd"]
    archs   = ["resnet20","resnet50","vgg19","mobilenetv2","efficientnet_b0","convnext_tiny"]
    sub = df[df["config_id"].isin(configs)].copy()
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Architecture} & \textbf{Method} & \textbf{Acc\textsubscript{clean} (\%)} & \textbf{mCE} & \textbf{ECE\textsubscript{clean}} & \textbf{Latency (ms)} & \textbf{Size (MB)} \\")
    lines.append(r"\midrule")
    for arch in archs:
        arch_sub = sub[sub["arch"] == arch].set_index("config_id")
        arch_label = ARCH_LABELS.get(arch, arch)
        lines.append(f"\\multirow{{6}}{{*}}{{{arch_label}}}")
        for cid in configs:
            if cid not in arch_sub.index:
                continue
            row = arch_sub.loc[cid]
            method = METHOD_LABELS.get(cid, cid)
            lines.append(f"  & {method} & {fmt(row['acc_clean'])} & {fmt(row['mce'])} & {fmt(row['ece_clean'],4)} & {fmt(row['latency_ms'])} & {fmt(row['size_mb'])} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_sparsity_sweep_table(csv_path, label, caption):
    df = pd.read_csv(csv_path)
    configs = ["baseline","pruning_20","pruning_50","pruning_70","pruning_90"]
    archs   = ["resnet20","resnet50","vgg19","mobilenetv2","efficientnet_b0","convnext_tiny"]
    sub = df[df["config_id"].isin(configs)].copy()
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Architecture} & \textbf{Sparsity} & \textbf{Acc\textsubscript{clean} (\%)} & \textbf{mCE} & \textbf{ECE\textsubscript{clean}} & \textbf{Size (MB)} \\")
    lines.append(r"\midrule")
    for arch in archs:
        arch_sub = sub[sub["arch"] == arch].set_index("config_id")
        arch_label = ARCH_LABELS.get(arch, arch)
        lines.append(f"\\multirow{{5}}{{*}}{{{arch_label}}}")
        for cid in configs:
            if cid not in arch_sub.index:
                continue
            row = arch_sub.loc[cid]
            method = METHOD_LABELS.get(cid, cid)
            lines.append(f"  & {method} & {fmt(row['acc_clean'])} & {fmt(row['mce'])} & {fmt(row['ece_clean'],4)} & {fmt(row['size_mb'])} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_kd_temp_table(csv_path, label, caption):
    df = pd.read_csv(csv_path)
    configs = ["baseline","kd_temp2","kd_temp4","kd_temp8"]
    archs   = ["resnet20","resnet50","vgg19","mobilenetv2","efficientnet_b0","convnext_tiny"]
    sub = df[df["config_id"].isin(configs)].copy()
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Architecture} & \textbf{Method} & \textbf{Acc\textsubscript{clean} (\%)} & \textbf{mCE} & \textbf{ECE\textsubscript{clean}} \\")
    lines.append(r"\midrule")
    for arch in archs:
        arch_sub = sub[sub["arch"] == arch].set_index("config_id")
        arch_label = ARCH_LABELS.get(arch, arch)
        lines.append(f"\\multirow{{4}}{{*}}{{{arch_label}}}")
        for cid in configs:
            if cid not in arch_sub.index:
                continue
            row = arch_sub.loc[cid]
            method = METHOD_LABELS.get(cid, cid)
            lines.append(f"  & {method} & {fmt(row['acc_clean'])} & {fmt(row['mce'])} & {fmt(row['ece_clean'],4)} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_phase2_table(csv_path, label, caption):
    df = pd.read_csv(csv_path)
    configs = [
        "baseline","quant_int8","sparsity","sparsity_quant_int8",
        "clustering","clustering_quant_int8","sparsity_clustering",
        "sparsity_clustering_quant_int8","qat_only","qat_int8",
        "cqat","cqat_int8","pqat","pqat_int8","pcqat","pcqat_int8"
    ]
    archs = ["resnet20","resnet50","vgg19","mobilenetv2","efficientnet_b0","convnext_tiny"]
    sub = df[df["config_id"].isin(configs)].copy()
    n = len(configs)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Architecture} & \textbf{Method} & \textbf{Acc\textsubscript{clean} (\%)} & \textbf{mCE} & \textbf{Size (MB)} \\")
    lines.append(r"\midrule")
    for arch in archs:
        arch_sub = sub[sub["arch"] == arch].set_index("config_id")
        arch_label = ARCH_LABELS.get(arch, arch)
        lines.append(f"\\multirow{{{n}}}{{*}}{{{arch_label}}}")
        for cid in configs:
            if cid not in arch_sub.index:
                continue
            row = arch_sub.loc[cid]
            method = METHOD_LABELS.get(cid, cid)
            lines.append(f"  & {method} & {fmt(row['acc_clean'])} & {fmt(row['mce'])} & {fmt(row['size_mb'])} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    c10  = "results/cifar10_results.csv"
    c100 = "results/cifar100_results.csv"
    os.makedirs("results/tables", exist_ok=True)

    tables = {
        "table_phase1_cifar10.tex":    make_phase1_table(c10,  "tab:phase1_cifar10",  "Phase 1 compression results on CIFAR-10."),
        "table_phase1_cifar100.tex":   make_phase1_table(c100, "tab:phase1_cifar100", "Phase 1 compression results on CIFAR-100."),
        "table_sparsity_cifar10.tex":  make_sparsity_sweep_table(c10,  "tab:sparsity_cifar10",  "Sparsity sweep results on CIFAR-10."),
        "table_sparsity_cifar100.tex": make_sparsity_sweep_table(c100, "tab:sparsity_cifar100", "Sparsity sweep results on CIFAR-100."),
        "table_kd_temp_cifar10.tex":   make_kd_temp_table(c10,  "tab:kd_temp_cifar10",  "KD temperature sweep on CIFAR-10."),
        "table_kd_temp_cifar100.tex":  make_kd_temp_table(c100, "tab:kd_temp_cifar100", "KD temperature sweep on CIFAR-100."),
        "table_phase2_cifar10.tex":    make_phase2_table(c10,  "tab:phase2_cifar10",  "Phase 2 (16-technique grid) results on CIFAR-10."),
        "table_phase2_cifar100.tex":   make_phase2_table(c100, "tab:phase2_cifar100", "Phase 2 (16-technique grid) results on CIFAR-100."),
    }

    for fname, content in tables.items():
        path = f"results/tables/{fname}"
        with open(path, "w") as f:
            f.write(content)
        print(f"Wrote {path}")
