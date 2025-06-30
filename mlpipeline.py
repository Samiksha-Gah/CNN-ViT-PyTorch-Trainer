

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import torchvision.datasets as D

import albumentations as A
from albumentations.pytorch import ToTensorV2

import optuna
from optuna.trial import Trial

from timm.models.vision_transformer import vit_base_patch16_224
from torch.utils.tensorboard import SummaryWriter


# ============ Custom Loss Combining Label Smoothing + Focal Loss ============
class LabelSmoothingFocalLoss(nn.Module):
    def __init__(self, smoothing=0.1, gamma=2.0):
        super().__init__()
        self.smoothing = smoothing
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, logits, target):
        n_classes = logits.size(-1)
        log_preds = self.log_softmax(logits)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        prob = log_preds.exp()
        focal_factor = (1 - prob) ** self.gamma
        loss = -true_dist * focal_factor * log_preds
        return loss.sum(dim=1).mean()


# ============ Hybrid CNN + ViT ============
class CNNViT(nn.Module):
    def __init__(self, n_classes=100, cnn_channels=[64, 128, 256], vit_pretrained=True):
        super().__init__()
        layers = []
        in_ch = 3
        for ch in cnn_channels:
            layers += [
                nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = ch
        self.cnn = nn.Sequential(*layers)
        self.adapter = nn.Conv2d(cnn_channels[-1], 3, kernel_size=1)
        self.vit = vit_base_patch16_224(pretrained=vit_pretrained)
        self.vit.head = nn.Linear(self.vit.head.in_features, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.adapter(x)
        x = nn.functional.interpolate(x, size=(224, 224),
                                      mode='bilinear', align_corners=False)
        return self.vit(x)


# ============ Data & Augmentations ============
def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.2, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        A.GridDropout(ratio=0.3, p=0.5),
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ])


class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, ds, transforms):
        self.ds = ds
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        img = np.array(img)
        img = self.transforms(image=img)['image']
        return img, label


# ============ Dataset Loader Factory ============
def load_datasets(cfg):
    if cfg["dataset"] == "cifar100":
        raw_train = D.CIFAR100(root=cfg["data_dir"], train=True, download=True)
        raw_val   = D.CIFAR100(root=cfg["data_dir"], train=False, download=True)
        train_ds = AlbumentationsDataset(raw_train, get_train_transforms())
        val_ds   = AlbumentationsDataset(raw_val,   get_val_transforms())
        n_classes = 100

    elif cfg["dataset"] == "imagenet":
        train_folder = os.path.join(cfg["data_dir"], "train")
        val_folder   = os.path.join(cfg["data_dir"], "val")
        train_ds = D.ImageFolder(
            train_folder,
            transform=lambda img: get_train_transforms()(image=np.array(img))["image"]
        )
        val_ds = D.ImageFolder(
            val_folder,
            transform=lambda img: get_val_transforms()(image=np.array(img))["image"]
        )
        n_classes = len(train_ds.classes)

    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset']}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=not cfg.get("distributed", False),
        num_workers=cfg.get("workers", 8),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("workers", 8),
        pin_memory=True
    )
    return train_loader, val_loader, n_classes


# ============ Training & Validation ============
def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    device, scheduler=None, accum_steps=1,
                    tb_writer=None, epoch=0):
    model.train()
    running_loss, running_acc = 0.0, 0
    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with autocast():
            logits = model(x)
            loss = criterion(logits, y) / accum_steps
        scaler.scale(loss).backward()
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step(epoch + i / len(loader))
        running_loss += loss.item() * accum_steps
        running_acc += (logits.argmax(-1) == y).sum().item()
    avg_loss = running_loss / len(loader.dataset)
    avg_acc = running_acc / len(loader.dataset)
    if tb_writer:
        tb_writer.add_scalar("train/loss", avg_loss, epoch)
        tb_writer.add_scalar("train/acc", avg_acc, epoch)
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, criterion, device, tb_writer=None, epoch=0):
    model.eval()
    total_loss, total_acc = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        total_acc += (logits.argmax(-1) == y).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_acc / len(loader.dataset)
    if tb_writer:
        tb_writer.add_scalar("val/loss", avg_loss, epoch)
        tb_writer.add_scalar("val/acc", avg_acc, epoch)
    return avg_loss, avg_acc


# ============ Main Training & Optuna Stub ============
def run_training(cfg, args):
    # distributed setup
    use_ddp = args.local_rank is not None and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        cfg["distributed"] = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare logging
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(os.path.join("runs", now)) if (not use_ddp or args.local_rank == 0) else None

    # datasets
    train_loader, val_loader, n_classes = load_datasets(cfg)
    if use_ddp:
        train_loader.sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset)
        val_loader.sampler   = torch.utils.data.distributed.DistributedSampler(val_loader.dataset, shuffle=False)

    # model
    model = CNNViT(n_classes=n_classes).to(device)
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # optimizer, scheduler, loss
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg["T_0"], T_mult=2)
    criterion = LabelSmoothingFocalLoss(smoothing=cfg["smoothing"], gamma=cfg["gamma"])
    scaler = GradScaler()

    # optionally resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)

    # training loop
    for epoch in range(start_epoch, cfg["epochs"]):
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, scheduler=scheduler,
            accum_steps=cfg["accum_steps"],
            tb_writer=tb_writer,
            epoch=epoch
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            tb_writer=tb_writer, epoch=epoch
        )
        if not use_ddp or args.local_rank == 0:
            print(f"[{epoch+1}/{cfg['epochs']}] train_loss={train_loss:.4f}, "
                  f"train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # checkpoint best
        if val_acc > best_acc:
            best_acc = val_acc
            if not use_ddp or args.local_rank == 0:
                save_path = os.path.join("runs", now, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "sched_state": scheduler.state_dict(),
                    "best_acc": best_acc
                }, save_path)

        # early stopping
        if epoch - cfg.get("early_stop_after", 10) > 0 and val_acc < best_acc:
            if not use_ddp or args.local_rank == 0:
                print("Validation stopped improving; early stopping.")
            break

    if tb_writer:
        tb_writer.close()


def objective(trial: Trial):
    cfg = {
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
        "smoothing": trial.suggest_float("smoothing", 0.0, 0.2),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "T_0": trial.suggest_int("T_0", 5, 20),
        "epochs": 30,
        "accum_steps": trial.suggest_int("accum_steps", 1, 4),
        "early_stop_after": 5,
        "dataset": trial.suggest_categorical("dataset", ["cifar100", "imagenet"]),
        "data_dir": "data",
        "workers": 8,
    }
    # We skip evaluate-only and resume in Optuna trials
    dummy_args = argparse.Namespace(local_rank=None, resume="", evaluate=False)
    run_training(cfg, dummy_args)
    return 0.0  # placeholder for real validation metric


def parse_args():
    p = argparse.ArgumentParser(description="Advanced ML pipeline demo")
    p.add_argument("--search", action="store_true", help="Do hyperparam search with Optuna")
    p.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    p.add_argument("--dataset", choices=["cifar100", "imagenet"], default="cifar100",
                   help="Which dataset to use")
    p.add_argument("--data-dir", type=str, default="data",
                   help="Root directory for CIFAR-100 or ImageNet-style folders")
    p.add_argument("--resume", type=str, default="",
                   help="Path to checkpoint to resume training from")
    p.add_argument("--evaluate", action="store_true",
                   help="Just run validation on the specified checkpoint and exit")
    p.add_argument("--local_rank", type=int, default=None,
                   help="Local process rank for distributed training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_cfg = {
        "lr": 3e-4,
        "batch_size": 128,
        "weight_decay": 1e-4,
        "smoothing": 0.1,
        "gamma": 2.0,
        "T_0": 10,
        "epochs": 50,
        "accum_steps": 2,
        "early_stop_after": 7,
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "workers": 8,
    }

    # Evaluate-only mode
    if args.evaluate:
        assert args.resume, "Must provide --resume path to checkpoint for evaluation"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.resume, map_location=device)
        model = CNNViT(n_classes=ckpt["model_state"][next(iter(ckpt["model_state"]))].shape[0]).to(device)
        model.load_state_dict(ckpt["model_state"])
        _, val_loader, _ = load_datasets(base_cfg)
        loss_fn = LabelSmoothingFocalLoss()
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        print(f"Evaluation → loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
        exit(0)

    # Hyperparameter search
    if args.search:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.trials)
        print("Best trial:", study.best_trial.params)

    # Regular training
    else:
        run_training(base_cfg, args)
