import argparse
import copy
import os

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import MultiStepLR

from models.registry import get_model, list_architectures
from data.loaders import get_loaders, get_calibration_loader
from compression.engine import compress


def get_device(cfg):
    if cfg.get("device", "auto") == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg["device"]


def save_model(model, path, is_quantized):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_quantized:
        try:
            torch.jit.script(model).save(path)
        except Exception:
            torch.jit.trace(model, torch.randn(1, 3, 32, 32)).save(path)
    else:
        torch.save(model.state_dict(), path)
    print(f"  Saved -> {path}  ({os.path.getsize(path)/1e6:.2f} MB)")


def ckpt_path(cfg, dataset, arch, config_id, is_quantized=False):
    ext = ".pt" if is_quantized else ".pth"
    return os.path.join(cfg["checkpoints_dir"], dataset, arch, f"{config_id}{ext}")


def train_baseline(arch, dataset, cfg, device):
    num_classes          = 10 if dataset == "cifar10" else 100
    t_cfg                = cfg["training"]
    train_loader, val_loader = get_loaders(dataset, cfg["data_dir"],
                                            batch_size=t_cfg["batch_size"],
                                            num_workers=cfg["num_workers"])
    model     = get_model(arch, num_classes=num_classes)
    if torch.cuda.device_count() > 1 and device == "cuda":
        print(f"  Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model     = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=t_cfg["lr"],
                                 momentum=t_cfg["momentum"], weight_decay=t_cfg["weight_decay"])
    scheduler = MultiStepLR(optimizer, milestones=t_cfg["lr_milestones"], gamma=t_cfg["lr_gamma"])
    criterion = nn.CrossEntropyLoss(label_smoothing=t_cfg.get("label_smoothing", 0.0))
    scaler    = torch.cuda.amp.GradScaler() if (cfg.get("mixed_precision") and device == "cuda") else None

    best_acc, best_state = 0.0, None
    for epoch in range(t_cfg["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                criterion(model(x), y).backward()
                optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            m_eval = model.module if isinstance(model, nn.DataParallel) else model
            m_eval.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    preds    = m_eval(x.to(device)).argmax(1).cpu()
                    correct += (preds == y).sum().item()
                    total   += y.size(0)
            acc = 100.0 * correct / total
            print(f"  [{arch}|{dataset}] epoch {epoch+1}/{t_cfg['epochs']}  val_acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc   = acc
                best_state = copy.deepcopy(m_eval.state_dict())

    m_final = model.module if isinstance(model, nn.DataParallel) else model
    m_final.load_state_dict(best_state)
    save_model(m_final, ckpt_path(cfg, dataset, arch, "baseline"), is_quantized=False)
    print(f"  Best val_acc: {best_acc:.2f}%")
    return m_final


def load_baseline(arch, dataset, cfg, device):
    path = ckpt_path(cfg, dataset, arch, "baseline")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline not found: {path}. Run --phase baseline first.")
    num_classes = 10 if dataset == "cifar10" else 100
    model       = get_model(arch, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)


def run_compression(arch, dataset, phase_configs, cfg, device):
    baseline             = load_baseline(arch, dataset, cfg, device)
    train_loader, val_loader = get_loaders(dataset, cfg["data_dir"],
                                            batch_size=cfg["training"]["batch_size"],
                                            num_workers=cfg["num_workers"])
    cal_loader           = get_calibration_loader(dataset, cfg["data_dir"], num_workers=cfg["num_workers"])
    for comp_cfg in phase_configs:
        cid    = comp_cfg["id"]
        method = comp_cfg.get("method", "none")
        if method == "none":
            continue
        print(f"\n  -- [{arch}|{dataset}] {comp_cfg['label']} --")
        compressed, is_q = compress(baseline, method, comp_cfg, train_loader, val_loader, cal_loader, device=device)
        save_model(compressed, ckpt_path(cfg, dataset, arch, cid, is_quantized=is_q), is_quantized=is_q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--arch",    default=None)
    parser.add_argument("--phase",   default="baseline",
                        choices=["baseline", "phase1", "phase2", "phase3", "all"])
    args   = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg)
    print(f"\nDevice: {device}")
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    archs = [args.arch] if args.arch else list_architectures()

    for arch in archs:
        print(f"\n{'='*60}\n  {arch} | {args.dataset}\n{'='*60}")

        if args.phase in ("baseline", "all"):
            path = ckpt_path(cfg, args.dataset, arch, "baseline")
            if os.path.exists(path):
                print(f"  [SKIP] Baseline exists: {path}")
            else:
                train_baseline(arch, args.dataset, cfg, device)

        for phase_key in ("phase1", "phase2", "phase3"):
            if args.phase in (phase_key, "all"):
                run_compression(arch, args.dataset, cfg.get(f"{phase_key}_configs", []), cfg, device)


if __name__ == "__main__":
    main()
