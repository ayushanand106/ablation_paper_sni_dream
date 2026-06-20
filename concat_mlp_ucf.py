#!/usr/bin/env python3
"""
Local runner for the reviewer-requested UCF101 capacity-control baseline:

    frozen DINOv2-Large + frozen BLIP-2 vision features + concat MLP head with approximately the same trainable
    parameter count as DREAM.

Expected local data layout:

    --dataset-dir should contain:
        classInd.txt
        trainlist01.txt
        trainlist02.txt
        trainlist03.txt

    --video-dir should contain class folders, e.g.:
        ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi

Example:

    python outputs/train_ucf101_dinov2_same_params_local.py \
      --dataset-dir /path/to/UCF101TrainTestSplits-RecognitionTask \
      --video-dir /path/to/UCF-101 \
      --batch-size 4 \
      --epochs 5
"""

import argparse
import gc
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as tv

try:
    import decord  # type: ignore
except ImportError:
    decord = None

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    from transformers import Dinov2Model, Blip2VisionModel
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: transformers. Install it with:\n"
        "  python -m pip install transformers\n"
    ) from exc


DREAM_TRAINABLE_PARAMS = 128_911_461
DEFAULT_HIDDEN_DIM = 10157


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to UCF101TrainTestSplits-RecognitionTask")
    parser.add_argument("--video-dir", required=True, help="Path to UCF-101 video folder")
    parser.add_argument("--output-dir", default="outputs/runs/ucf_concat_dinov2_blip2_same_params_mlp")
    parser.add_argument("--dino-model-name", default="facebook/dinov2-large")
    parser.add_argument("--blip-model-name", default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-classes", type=int, default=101)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None, help="Debug only: limit dataset size")
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


class UCFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, video_dir, subset, video_list_file, frames_per_clip=8, limit_samples=None):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.video_dir = Path(video_dir)
        self.subset = subset
        self.video_list = []
        self.labels = []

        for i in [1, 2, 3]:
            file_path = self.dataset_dir / f"{video_list_file}{i}.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing split file: {file_path}")
            with file_path.open("r") as video_names_file:
                if self.subset == "train":
                    pairs = [line.strip().split() for line in video_names_file if line.strip()]
                    tempvideo_list, templabels = zip(*pairs)
                    self.video_list.extend(tempvideo_list)
                    self.labels.extend(templabels)
                else:
                    current_videos = [line.strip() for line in video_names_file if line.strip()]
                    self.video_list.extend(current_videos)
                    self.labels.extend([None] * len(current_videos))

        if limit_samples is not None:
            self.video_list = self.video_list[:limit_samples]
            self.labels = self.labels[:limit_samples]

        self.frames_per_clip = frames_per_clip
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])

        class_file = self.dataset_dir / "classInd.txt"
        if class_file.exists():
            with class_file.open("r") as class_indices:
                self.class_map = {line.split()[1]: int(line.split()[0]) for line in class_indices}
        else:
            self.class_map = None

    def __len__(self):
        return len(self.video_list)

    def _read_video_frames(self, video_path):
        if decord is not None:
            vid = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
            nframes = len(vid)

            if nframes <= self.frames_per_clip:
                idxs = np.arange(self.frames_per_clip) % nframes
            else:
                idxs = np.linspace(0, nframes - 1, self.frames_per_clip).astype(int)

            return [vid[i].asnumpy() for i in idxs]

        if cv2 is None:
            raise RuntimeError(
                "Missing video backend. Install opencv-python on macOS/arm64, "
                "or decord on a supported Linux/Windows platform."
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        decoded_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        nframes = len(decoded_frames)
        if nframes <= 0:
            raise RuntimeError(f"Could not decode any frames from video: {video_path}")

        if nframes <= self.frames_per_clip:
            idxs = np.arange(self.frames_per_clip) % nframes
        else:
            idxs = np.linspace(0, nframes - 1, self.frames_per_clip).astype(int)

        return [decoded_frames[i] for i in idxs]

    def __getitem__(self, idx):
        videoname = self.video_list[idx]
        video_path = self.video_dir / videoname
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video: {video_path}")

        frames = self._read_video_frames(video_path)
        imgs = [self.transform(Image.fromarray(frame).convert("RGB")) for frame in frames]
        imgs = torch.stack(imgs, dim=1)  # C, T, H, W

        if self.subset == "train":
            label = int(self.labels[idx]) - 1
        else:
            if self.class_map is None:
                raise RuntimeError("classInd.txt is required for test labels.")
            label = self.class_map[videoname.split("/")[0]] - 1

        return imgs, label


class ConcatDinoBLIPSameParamsMLP(nn.Module):
    def __init__(
        self,
        n_class=101,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        dino_model_name="facebook/dinov2-large",
        blip_model_name="Salesforce/blip2-flan-t5-xl",
        cache_dir=None,
    ):
        super().__init__()

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self.dino = Dinov2Model.from_pretrained(dino_model_name, **kwargs)
        self.blip = Blip2VisionModel.from_pretrained(blip_model_name, **kwargs)
        self.dino.eval()
        self.blip.eval()

        for param in self.dino.parameters():
            param.requires_grad = False
        for param in self.blip.parameters():
            param.requires_grad = False

        self.dino_dim = self.dino.config.hidden_size
        self.blip_dim = getattr(self.blip.config, "hidden_size", 1408)
        self.feature_dim = self.dino_dim + self.blip_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_class),
        )

    def forward(self, inp):
        bsz, channels, frames, height, width = inp.shape
        inp = inp.transpose(1, 2).reshape(bsz * frames, channels, height, width)

        with torch.no_grad():
            dino_feats = self.dino(inp).last_hidden_state[:, 0, :]
            blip_feats = self.blip(pixel_values=inp).last_hidden_state[:, 0, :]

        feats = torch.cat([dino_feats, blip_feats], dim=-1)
        feats = feats.reshape(bsz, frames, -1).mean(dim=1)
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


def move_batch_to_device(video_data, labels, device):
    return video_data.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def train_step(model, loader, epoch, optimizer, scheduler, loss_criterion, device):
    model.train()
    model.dino.eval()
    model.blip.eval()

    total_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    pbar = tqdm(loader, desc=f"[Train Epoch {epoch}]", leave=False)
    for video_data, labels in pbar:
        video_data, labels = move_batch_to_device(video_data, labels, device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(video_data)
        loss = loss_criterion(prediction, labels)

        top1, top5 = top_k_accuracy(prediction, labels)
        total_loss += loss.item()
        top1_sum += top1.item()
        top5_sum += top5.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix({"loss": loss.item(), "top1": top1.item(), "top5": top5.item()})

        del video_data, labels, prediction, loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    top1_accuracy = top1_sum / len(loader)
    top5_accuracy = top5_sum / len(loader)
    avg_loss = total_loss / len(loader)
    print(f"[Train Epoch {epoch}] Top-1 Acc: {top1_accuracy:.2f}% | Top-5 Acc: {top5_accuracy:.2f}% | Loss: {avg_loss:.4f}")
    return top1_accuracy, top5_accuracy, avg_loss


@torch.no_grad()
def val_step(model, loader, epoch, loss_criterion, device):
    model.eval()

    total_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    pbar = tqdm(loader, desc=f"[Val Epoch {epoch}]", leave=False)
    for video_data, labels in pbar:
        video_data, labels = move_batch_to_device(video_data, labels, device)
        prediction = model(video_data)
        loss = loss_criterion(prediction, labels)

        top1, top5 = top_k_accuracy(prediction, labels)
        total_loss += loss.item()
        top1_sum += top1.item()
        top5_sum += top5.item()

        pbar.set_postfix({"loss": loss.item(), "top1": top1.item(), "top5": top5.item()})

        del video_data, labels, prediction, loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    top1_accuracy = top1_sum / len(loader)
    top5_accuracy = top5_sum / len(loader)
    avg_loss = total_loss / len(loader)
    print(f"[Val Epoch {epoch}] Top-1 Acc: {top1_accuracy:.2f}% | Top-5 Acc: {top5_accuracy:.2f}% | Loss: {avg_loss:.4f}")
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

    train_val_data = UCFDataset(
        args.dataset_dir,
        args.video_dir,
        "train",
        "trainlist0",
        args.frames_per_clip,
        limit_samples=args.limit_samples,
    )
    val_len = int(args.val_ratio * len(train_val_data))
    train_len = len(train_val_data) - val_len
    train_data, val_data = random_split(
        train_val_data,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
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

    model = ConcatDinoBLIPSameParamsMLP(
        args.num_classes,
        hidden_dim=args.hidden_dim,
        dino_model_name=args.dino_model_name,
        blip_model_name=args.blip_model_name,
        cache_dir=args.cache_dir,
    ).to(device)

    trainable_params = count_trainable_params(model)
    run_info = {
        "device": str(device),
        "dataset_dir": args.dataset_dir,
        "video_dir": args.video_dir,
        "dino_model_name": args.dino_model_name,
        "blip_model_name": args.blip_model_name,
        "frames_per_clip": args.frames_per_clip,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
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
            checkpoint_path = checkpoint_dir / f"ucf_concat_dinov2_blip2_same_params_mlp_{timestamp}_epoch_{epoch}.pth"
            print(f"Saving model at epoch {epoch}: {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Best validation Top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()
