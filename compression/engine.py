"""
compression/engine.py
Unified compression engine. All methods in one place.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


# ── Pruning ───────────────────────────────────────────────────────

def _prunable_params(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def apply_unstructured_pruning(model, sparsity):
    prune.global_unstructured(_prunable_params(model), pruning_method=prune.L1Unstructured, amount=sparsity)
    return model


def apply_structured_pruning(model, sparsity):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 1:
            n_prune = max(1, int(module.out_channels * sparsity))
            prune.ln_structured(module, name="weight", amount=n_prune, n=1, dim=0)
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=sparsity)
    return model


def remove_masks(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(m, "weight")
            except ValueError:
                pass
    return model


def get_sparsity(model):
    total = zeros = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total += m.weight.numel()
            zeros += (m.weight == 0).sum().item()
    return zeros / total if total > 0 else 0.0


# ── Clustering ────────────────────────────────────────────────────

def apply_clustering(model, n_clusters=16):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                w = m.weight.data.view(-1)
                idx = torch.randperm(len(w))[:n_clusters]
                centroids = w[idx].clone()
                for _ in range(50):
                    dists  = (w.unsqueeze(1) - centroids.unsqueeze(0)).abs()
                    assign = dists.argmin(dim=1)
                    for k in range(n_clusters):
                        mask = assign == k
                        if mask.any():
                            centroids[k] = w[mask].mean()
                dists  = (w.unsqueeze(1) - centroids.unsqueeze(0)).abs()
                assign = dists.argmin(dim=1)
                m.weight.data = centroids[assign].view(m.weight.shape)
    return model


# ── Quantization ──────────────────────────────────────────────────

def _prepare_ptq(model):
    m = copy.deepcopy(model).cpu().eval()
    m.qconfig = torch.quantization.get_default_qconfig("x86")
    torch.quantization.prepare(m, inplace=True)
    return m


def _prepare_qat(model):
    m = copy.deepcopy(model).cpu().train()
    m.qconfig = torch.quantization.get_default_qat_qconfig("x86")
    torch.quantization.prepare_qat(m, inplace=True)
    return m


def _calibrate(model, loader, n_batches=32, device="cpu"):
    model.eval().to(device)
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(x.to(device))


def _convert_int8(model):
    model.cpu().eval()
    torch.quantization.convert(model, inplace=True)
    return model


# ── Training loops ────────────────────────────────────────────────

def _train_loop(model, train_loader, val_loader, epochs, lr,
                lr_milestones=None, lr_gamma=0.1, device="cpu", scheduler_type="cosine"):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = MultiStepLR(optimizer, milestones=lr_milestones or [60, 80], gamma=lr_gamma)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        if epoch == max(1, epochs // 2):
            try:
                model.apply(torch.quantization.disable_observer)
                model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            except Exception:
                pass
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = _quick_acc(model, val_loader, device)
            print(f"    epoch [{epoch+1}/{epochs}]  val_acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
    if best_state:
        model.load_state_dict(best_state)
    return model


def _quick_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


# ── Knowledge Distillation ────────────────────────────────────────

def _kd_loss(s_logits, t_logits, labels, T, alpha):
    soft = F.kl_div(
        F.log_softmax(s_logits / T, 1),
        F.softmax(t_logits / T, 1),
        reduction="batchmean",
    ) * T ** 2
    hard = F.cross_entropy(s_logits, labels, label_smoothing=0.1)
    return alpha * soft + (1 - alpha) * hard


def train_kd(student, teacher, train_loader, val_loader,
             epochs=100, lr=0.1, lr_milestones=None, lr_gamma=0.1,
             temperature=4.0, alpha=0.7, device="cpu"):
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones or [60, 80], gamma=lr_gamma)
    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            loss = _kd_loss(student(x), t_logits, y, temperature, alpha)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = _quick_acc(student, val_loader, device)
            print(f"    KD epoch [{epoch+1}/{epochs}]  val_acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(student.state_dict())
    if best_state:
        student.load_state_dict(best_state)
    return student


# ── Main compress() entry point ───────────────────────────────────

def compress(model, method, cfg, train_loader, val_loader, calibration_loader, device="cuda"):
    m = copy.deepcopy(model)

    if method == "none":
        return m.to(device), False

    if method == "fp16":
        return m.half().to(device), False

    if method == "pruning":
        apply_unstructured_pruning(m, cfg["sparsity"])
        print(f"  Actual sparsity: {get_sparsity(m):.1%}")
        m = _train_loop(m, train_loader, val_loader, cfg["fine_tune_epochs"], cfg["fine_tune_lr"], device=device)
        remove_masks(m)
        return m, False

    if method == "structured_pruning":
        apply_structured_pruning(m, cfg["sparsity"])
        m = _train_loop(m, train_loader, val_loader, cfg["fine_tune_epochs"], cfg["fine_tune_lr"], device=device)
        remove_masks(m)
        return m, False

    if method == "ptq":
        m.cpu()
        prepared = _prepare_ptq(m)
        _calibrate(prepared, calibration_loader, cfg.get("calibration_batches", 32))
        return _convert_int8(prepared), True

    if method in ("qat", "qat_then_convert"):
        m.cpu()
        qat_m = _prepare_qat(m)
        qat_m = _train_loop(qat_m, train_loader, val_loader, cfg["epochs"], cfg["lr"], device="cpu")
        return _convert_int8(qat_m), True

    if method == "pruning_then_ptq":
        apply_unstructured_pruning(m, cfg["sparsity"])
        m = _train_loop(m, train_loader, val_loader, cfg["fine_tune_epochs"], cfg["fine_tune_lr"], device=device)
        remove_masks(m)
        m.cpu()
        prepared = _prepare_ptq(m)
        _calibrate(prepared, calibration_loader, cfg.get("calibration_batches", 32))
        return _convert_int8(prepared), True

    if method == "pruning_then_qat":
        apply_unstructured_pruning(m, cfg["sparsity"])
        m = _train_loop(m, train_loader, val_loader, cfg["fine_tune_epochs"], cfg["fine_tune_lr"], device=device)
        remove_masks(m)
        m.cpu()
        qat_m = _prepare_qat(m)
        qat_m = _train_loop(qat_m, train_loader, val_loader, cfg["qat_epochs"], cfg["qat_lr"], device="cpu")
        return _convert_int8(qat_m), True

    if method in ("pruning_then_kd", "kd"):
        student = copy.deepcopy(model)
        apply_unstructured_pruning(student, cfg["sparsity"])
        remove_masks(student)
        student = train_kd(
            student, model, train_loader, val_loader,
            epochs=cfg["epochs"], lr=cfg["lr"],
            lr_milestones=cfg.get("lr_milestones", [60, 80]),
            lr_gamma=cfg.get("lr_gamma", 0.1),
            temperature=cfg["kd_temperature"], alpha=cfg["kd_alpha"],
            device=device,
        )
        return student, False

    if method == "clustering":
        apply_clustering(m, cfg.get("n_clusters", 16))
        return m.to(device), False

    if method == "clustering_then_ptq":
        apply_clustering(m, cfg.get("n_clusters", 16))
        m.cpu()
        prepared = _prepare_ptq(m)
        _calibrate(prepared, calibration_loader, cfg.get("calibration_batches", 32))
        return _convert_int8(prepared), True

    if method in ("sparsity_preserving_clustering", "sparsity_preserving_clustering_then_ptq"):
        apply_unstructured_pruning(m, cfg["sparsity"])
        m = _train_loop(m, train_loader, val_loader, cfg.get("fine_tune_epochs", 20), cfg.get("fine_tune_lr", 0.01), device=device)
        remove_masks(m)
        apply_clustering(m, cfg.get("n_clusters", 16))
        if method.endswith("_ptq"):
            m.cpu()
            prepared = _prepare_ptq(m)
            _calibrate(prepared, calibration_loader, cfg.get("calibration_batches", 32))
            return _convert_int8(prepared), True
        return m.to(device), False

    if method in ("cqat", "cqat_then_convert"):
        apply_clustering(m, cfg.get("n_clusters", 16))
        m.cpu()
        qat_m = _prepare_qat(m)
        qat_m = _train_loop(qat_m, train_loader, val_loader, cfg["epochs"], cfg["lr"], device="cpu")
        return _convert_int8(qat_m), True

    if method in ("pqat", "pqat_then_convert"):
        apply_unstructured_pruning(m, cfg["sparsity"])
        m = _train_loop(m, train_loader, val_loader, cfg.get("fine_tune_epochs", 10), cfg.get("fine_tune_lr", 0.01), device=device)
        remove_masks(m)
        m.cpu()
        qat_m = _prepare_qat(m)
        qat_m = _train_loop(qat_m, train_loader, val_loader, cfg["epochs"], cfg["lr"], device="cpu")
        return _convert_int8(qat_m), True

    if method in ("pcqat", "pcqat_then_convert"):
        apply_unstructured_pruning(m, cfg["sparsity"])
        m = _train_loop(m, train_loader, val_loader, cfg.get("fine_tune_epochs", 10), cfg.get("fine_tune_lr", 0.01), device=device)
        remove_masks(m)
        apply_clustering(m, cfg.get("n_clusters", 16))
        m.cpu()
        qat_m = _prepare_qat(m)
        qat_m = _train_loop(qat_m, train_loader, val_loader, cfg["epochs"], cfg["lr"], device="cpu")
        return _convert_int8(qat_m), True

    raise ValueError(f"Unknown compression method: '{method}'")
