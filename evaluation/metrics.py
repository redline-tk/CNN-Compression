import os
import time
import numpy as np
import torch


@torch.no_grad()
def collect(model, loader, device="cpu"):
    all_probs, all_labels = [], []
    model.eval().to(device)
    for x, y in loader:
        all_probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        all_labels.append(y)
    return torch.cat(all_probs), torch.cat(all_labels)


def accuracy(probs, labels):
    return 100.0 * (probs.argmax(1) == labels).float().mean().item()


def top5_accuracy(probs, labels):
    return 100.0 * probs.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).any(1).float().mean().item()


def ece(probs, labels, n_bins=15):
    confs, preds = probs.max(dim=1)
    accs         = preds.eq(labels).float()
    bins         = torch.linspace(0, 1, n_bins + 1)
    ece_val      = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() > 0:
            ece_val += mask.float().mean().item() * abs(accs[mask].mean().item() - confs[mask].mean().item())
    return ece_val


def compute_baseline_errors(model, corruption_loader_fn, corruptions, severities=None, device="cpu"):
    severities = severities or [1, 2, 3, 4, 5]
    errors     = {}
    for c in corruptions:
        for s in severities:
            probs, labels  = collect(model, corruption_loader_fn(c, s), device=device)
            errors[(c, s)] = 1.0 - accuracy(probs, labels) / 100.0
    return errors


def corruption_error(model, corruption_loader_fn, baseline_errors, corruptions, severities=None, device="cpu"):
    severities = severities or [1, 2, 3, 4, 5]
    ce_per     = {}
    for c in corruptions:
        model_sum = baseline_sum = 0.0
        for s in severities:
            probs, labels = collect(model, corruption_loader_fn(c, s), device=device)
            err           = 1.0 - accuracy(probs, labels) / 100.0
            model_sum    += err
            baseline_sum += baseline_errors.get((c, s), err)
        ce_per[c] = model_sum / baseline_sum if baseline_sum > 0 else 1.0
    return ce_per, 100.0 * np.mean(list(ce_per.values()))


def measure_latency(model, input_shape=(1, 3, 32, 32), n_warmup=50, n_runs=500, device="cpu"):
    model.eval().to(device)
    dummy = torch.randn(input_shape).to(device)
    for _ in range(n_warmup):
        with torch.no_grad():
            model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def model_size_mb(path):
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else float("nan")


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def compression_ratio(original_mb, compressed_mb):
    return original_mb / compressed_mb if compressed_mb > 0 else float("nan")
