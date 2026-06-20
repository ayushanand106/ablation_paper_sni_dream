#!/usr/bin/env python3
"""
Local runner for the reviewer-requested CIFAR-100 capacity-control baseline:

    frozen DINOv2-Large + MLP head with approximately the same trainable
    parameter count as DREAM.

Expected local data layout:

    --data-dir can be any writable directory. torchvision will download
    CIFAR-100 there when --download is passed.

Example:

    python dino_training_c100_mlp.py \
      --data-dir /path/to/cifar100 \
      --download \
      --batch-size 32 \
      --epochs 20
"""

import argparse
import gc
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tv

try:
    from transformers import Dinov2Model
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: transformers. Install it with:\n"
        "  python -m pip install transformers\n"
    ) from exc


DREAM_TRAINABLE_PARAMS = 128_911_461
DEFAULT_HIDDEN_DIM = 10805


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path where CIFAR-100 is stored/downloaded")
    parser.add_argument("--output-dir", default="outputs/runs/c100_dinov2_large_same_params_mlp")
    parser.add_argument("--model-name", default="facebook/dinov2-large")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None, help="Debug only: limit training dataset size")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transform():
    return tv.Compose([
        tv.Resize((224, 224)),
        tv.ToTensor(),
        tv.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])


def maybe_limit_dataset(dataset, limit_samples):
    if limit_samples is None:
        return dataset
    limit = min(limit_samples, len(dataset))
    return torch.utils.data.Subset(dataset, range(limit))


class DINOv2LargeSameParamsMLP(nn.Module):
    def __init__(self, n_class=100, hidden_dim=DEFAULT_HIDDEN_DIM, model_name="facebook/dinov2-large", cache_dir=None):
        super().__init__()

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self.dino = Dinov2Model.from_pretrained(model_name, **kwargs)
        self.dino.eval()

        for param in self.dino.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_class),
        )

    def forward(self, inp):
        with torch.no_grad():
            feats = self.dino(inp).last_hidden_state[:, 0, :]
        return self.mlp(feats)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def top_k_accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def move_batch_to_device(image_data, labels, device):
    return image_data.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def train_step(model, loader, epoch, optimizer, scheduler, loss_criterion, device):
    model.train()
    model.dino.eval()

    total_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    pbar = tqdm(loader, desc=f"[Train Epoch {epoch}]", leave=False)
    for image_data, labels in pbar:
        image_data, labels = move_batch_to_device(image_data, labels, device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(image_data)
        loss = loss_criterion(prediction, labels)

        top1, top5 = top_k_accuracy(prediction, labels)
        total_loss += loss.item()
        top1_sum += top1.item()
        top5_sum += top5.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix({"loss": loss.item(), "top1": top1.item(), "top5": top5.item()})

        del image_data, labels, prediction, loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    top1_accuracy = top1_sum / len(loader)
    top5_accuracy = top5_sum / len(loader)
    avg_loss = total_loss / len(loader)
    print(f"[Train Epoch {epoch}] Top-1 Acc: {top1_accuracy:.2f}% | Top-5 Acc: {top5_accuracy:.2f}% | Loss: {avg_loss:.4f}")
    return top1_accuracy, top5_accuracy, avg_loss


@torch.no_grad()
def val_step(model, loader, epoch, loss_criterion, device, split_name="Val"):
    model.eval()

    total_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    pbar = tqdm(loader, desc=f"[{split_name} Epoch {epoch}]", leave=False)
    for image_data, labels in pbar:
        image_data, labels = move_batch_to_device(image_data, labels, device)
        prediction = model(image_data)
        loss = loss_criterion(prediction, labels)

        top1, top5 = top_k_accuracy(prediction, labels)
        total_loss += loss.item()
        top1_sum += top1.item()
        top5_sum += top5.item()

        pbar.set_postfix({"loss": loss.item(), "top1": top1.item(), "top5": top5.item()})

        del image_data, labels, prediction, loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    top1_accuracy = top1_sum / len(loader)
    top5_accuracy = top5_sum / len(loader)
    avg_loss = total_loss / len(loader)
    print(f"[{split_name} Epoch {epoch}] Top-1 Acc: {top1_accuracy:.2f}% | Top-5 Acc: {top5_accuracy:.2f}% | Loss: {avg_loss:.4f}")
    return top1_accuracy, top5_accuracy, avg_loss


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    pin_memory = not args.no_pin_memory and device.type == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform()
    train_data = torchvision.datasets.CIFAR100(
        root=args.data_dir,
        train=True,
        transform=transform,
        download=args.download,
    )
    train_data = maybe_limit_dataset(train_data, args.limit_samples)
    val_data = torchvision.datasets.CIFAR100(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=args.download,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    model = DINOv2LargeSameParamsMLP(
        args.num_classes,
        hidden_dim=args.hidden_dim,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
    ).to(device)

    trainable_params = count_trainable_params(model)
    run_info = {
        "device": str(device),
        "data_dir": args.data_dir,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_classes": args.num_classes,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "validation_split": "CIFAR100 train=False",
        "trainable_params": trainable_params,
        "dream_trainable_params": DREAM_TRAINABLE_PARAMS,
        "difference_from_dream": trainable_params - DREAM_TRAINABLE_PARAMS,
        "relative_difference_pct": (trainable_params - DREAM_TRAINABLE_PARAMS) / DREAM_TRAINABLE_PARAMS * 100,
    }
    print(json.dumps(run_info, indent=2))

    print("Trainable parameter names:")
    for name, params in model.named_parameters():
        if params.requires_grad:
            print(name, tuple(params.shape))

    with (output_dir / "run_info.json").open("w") as f:
        json.dump(run_info, f, indent=2)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_top1 = float("-inf")
    history = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for epoch in range(1, args.epochs + 1):
        train_top1, train_top5, train_loss = train_step(
            model, train_loader, epoch, optimizer, scheduler, loss_criterion, device
        )
        val_top1, val_top5, val_loss = val_step(model, val_loader, epoch, loss_criterion, device)

        row = {
            "epoch": epoch,
            "train_top1": train_top1,
            "train_top5": train_top5,
            "train_loss": train_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "val_loss": val_loss,
            "trainable_params": trainable_params,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        with (output_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

        if val_top1 > best_top1:
            best_top1 = val_top1
            checkpoint_path = checkpoint_dir / f"c100_dinov2_large_same_params_mlp_{timestamp}_epoch_{epoch}.pth"
            print(f"Best validation Top-1 improved to {best_top1:.2f}% at epoch {epoch}")
            print(f"Saving best model: {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

    best_metrics = {"best_val_top1": best_top1}
    with (output_dir / "best_metrics.json").open("w") as f:
        json.dump(best_metrics, f, indent=2)

    print(json.dumps(best_metrics, indent=2))
    print(f"Best validation Top-1 on CIFAR-100 test set: {best_top1:.2f}%")


if __name__ == "__main__":
    main()
