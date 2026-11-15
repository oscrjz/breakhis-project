"""Train and evaluate one BreakHis experiment.

Examples
  python train.py --data-dir /path/to/BreaKHis --model mobilenet_v2 --run-name e1_mobilenet
  python train.py --data-dir ... --model resnet50 --run-name e2_resnet50
  python train.py --data-dir ... --model mobilenet_v2 --loss weighted --run-name e3_weighted
  python train.py --data-dir ... --model mobilenet_v2 --teacher checkpoints/e2_resnet50.pt --run-name e7_distill

Each run writes results/<run-name>.json (config + metrics) and checkpoints/<run-name>.pt.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from breakhis.data import (BreakHisDataset, build_index, build_transforms,
                           find_official_folds, make_split)
from breakhis.metrics import compute_metrics
from breakhis.models import SUPPORTED, build_model, count_params


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder containing the BreakHis images (searched recursively)")
    p.add_argument("--model", default="mobilenet_v2", choices=SUPPORTED)
    p.add_argument("--no-pretrained", action="store_true", help="start from random weights instead of ImageNet")
    p.add_argument("--split", default="patient", choices=["patient", "image"], help="image = random split, RQ3 only")
    p.add_argument("--fold", type=int, default=None, help="use official Folds.csv fold 1-5 instead of our own split")
    p.add_argument("--magnification", type=int, default=None, choices=[40, 100, 200, 400],
                   help="train on one zoom level only; default pools all four")
    p.add_argument("--loss", default="ce", choices=["ce", "weighted"], help="weighted = class-weighted cross-entropy")
    p.add_argument("--sampler", default="none", choices=["none", "oversample"], help="oversample the minority class")
    p.add_argument("--stain-aug", action="store_true", help="HED stain augmentation")
    p.add_argument("--teacher", default=None, help="checkpoint of a trained teacher for knowledge distillation")
    p.add_argument("--kd-alpha", type=float, default=0.5, help="weight of the hard-label loss when distilling")
    p.add_argument("--kd-temp", type=float, default=4.0, help="softmax temperature when distilling")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="use only N images, for quick debugging")
    p.add_argument("--run-name", required=True)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--ckpt-dir", default="checkpoints")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, idxs = [], []
    for x, _, i in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits, 1)[:, 1].cpu())
        idxs.append(i)
    return torch.cat(probs).numpy(), torch.cat(idxs).numpy()


def evaluate(model, loader, df, device):
    prob, idx = predict(model, loader, device)
    sub = df.iloc[idx]
    return compute_metrics(sub["label"].to_numpy(), prob, sub["patient"].to_numpy())


@torch.no_grad()
def cpu_ms_per_image(model, img_size, n=30):
    """Single-image CPU latency, for RQ1."""
    m = model.to("cpu").eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(3):
        m(x)
    t = time.perf_counter()
    for _ in range(n):
        m(x)
    return (time.perf_counter() - t) / n * 1000


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir, ckpt_dir = Path(args.out_dir), Path(args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    df = build_index(args.data_dir)
    if args.magnification:
        df = df[df["magnification"] == args.magnification].reset_index(drop=True)
    if args.limit:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.limit // 2), random_state=args.seed)).reset_index(drop=True)
    folds = find_official_folds(args.data_dir) if args.fold else None
    df["split"] = make_split(df, args.split, seed=args.seed, fold=args.fold, folds=folds).to_numpy()
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if args.split == "patient":
        sets = [set(parts[s]["patient"]) for s in parts]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "patient leak between splits"
    for s, d in parts.items():
        print(f"{s:5s}: {len(d):5d} images, {d['patient'].nunique():3d} patients, "
              f"{(d['label'] == 0).sum()} benign / {(d['label'] == 1).sum()} malignant")

    train_ds = BreakHisDataset(parts["train"], build_transforms(args.img_size, True, args.stain_aug))
    eval_tf = build_transforms(args.img_size, False)
    counts = np.bincount(parts["train"]["label"], minlength=2)
    sampler = None
    if args.sampler == "oversample":
        w = 1.0 / counts[parts["train"]["label"].to_numpy()]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device == "cuda")
    train_dl = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
    val_dl = DataLoader(BreakHisDataset(parts["val"], eval_tf), **kw)
    test_dl = DataLoader(BreakHisDataset(parts["test"], eval_tf), **kw)

    # ---------- model ----------
    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    class_w = None
    if args.loss == "weighted":
        class_w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float, device=device)
    teacher = None
    if args.teacher:
        ck = torch.load(args.teacher, map_location=device)
        teacher = build_model(ck["model_name"], pretrained=False).to(device)
        teacher.load_state_dict(ck["state_dict"])
        teacher.eval()
        for prm in teacher.parameters():
            prm.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    # ---------- train ----------
    best, best_epoch, history = -1.0, -1, []
    ckpt_path = ckpt_dir / f"{args.run_name}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=class_w)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    T = args.kd_temp
                    kd = F.kl_div(F.log_softmax(logits / T, 1), F.softmax(t_logits / T, 1),
                                  reduction="batchmean") * T * T
                    loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
        sched.step()
        val = evaluate(model, val_dl, parts["val"], device)
        history.append({"epoch": epoch, "train_loss": run_loss / len(train_ds), **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {epoch:2d}  loss {run_loss / len(train_ds):.4f}  val bal_acc {val['balanced_accuracy']:.3f}  "
              f"val auroc {val['auroc']:.3f}  ({time.time() - t0:.0f}s)")
        if val["balanced_accuracy"] > best:
            best, best_epoch = val["balanced_accuracy"], epoch
            torch.save({"model_name": args.model, "state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)

    # ---------- test with the best validation checkpoint ----------
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    test = evaluate(model, test_dl, parts["test"], device)
    result = {
        "run_name": args.run_name,
        "config": vars(args),
        "best_epoch": best_epoch,
        "params": count_params(model),
        "cpu_ms_per_image": cpu_ms_per_image(model, args.img_size),
        "n_images": {s: len(d) for s, d in parts.items()},
        "n_patients": {s: int(d["patient"].nunique()) for s, d in parts.items()},
        "test": test,
        "history": history,
    }
    (out_dir / f"{args.run_name}.json").write_text(json.dumps(result, indent=2))
    print("TEST", json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
