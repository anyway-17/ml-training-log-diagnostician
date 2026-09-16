import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class SimpleCNN(nn.Module):
    def __init__(self, freeze_conv1: bool = False, freeze_conv2: bool = False, init_scheme: str = "default", dropout_rate: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout_rate)

        if init_scheme == "bad":
            for m in [self.conv1, self.conv2, self.fc1, self.fc2]:
                nn.init.normal_(m.weight, mean=0.0, std=5.0)
                nn.init.zeros_(m.bias)

        if freeze_conv1:
            for p in self.conv1.parameters():
                p.requires_grad = False
        if freeze_conv2:
            for p in self.conv2.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_grad_norm(model) -> float:
    """L2 norm across all gradients — used to detect vanishing/exploding gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_dataloaders(batch_size: int, train_subset_frac: float = 1.0, label_noise_frac: float = 0.0, seed: int = 42):
    transform = transforms.Compose([transforms.ToTensor()])

    full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    # Optional label noise injection (for the label-noise failure mode)
    if label_noise_frac > 0.0:
        rng = np.random.RandomState(seed)
        targets = full_train.targets.clone()
        n = len(targets)
        n_flip = int(n * label_noise_frac)
        flip_idx = rng.choice(n, size=n_flip, replace=False)
        n_classes = 10
        for idx in flip_idx:
            orig = targets[idx].item()
            choices = [c for c in range(n_classes) if c != orig]
            targets[idx] = rng.choice(choices)
        full_train.targets = targets

    # Split into train/val
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Optional train subset shrink (for the overfitting failure mode)
    if train_subset_frac < 1.0:
        subset_size = int(len(train_set) * train_subset_frac)
        train_set, _ = random_split(
            train_set, [subset_size, len(train_set) - subset_size],
            generator=torch.Generator().manual_seed(seed)
        )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


##Training LOOP:

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        train_subset_frac=args.train_subset_frac,
        label_noise_frac=args.label_noise_frac,
        seed=args.seed,
    )

    model = SimpleCNN(
        freeze_conv1=args.freeze_conv1,
        freeze_conv2=args.freeze_conv2,
        init_scheme=args.init_scheme,
        dropout_rate=args.dropout_rate,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    epoch_logs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        epoch_grad_norms = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()

            epoch_grad_norms.append(compute_grad_norm(model))
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        avg_grad_norm = float(np.mean(epoch_grad_norms))
        current_lr = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_acc": round(val_acc, 6),
            "grad_norm": round(avg_grad_norm, 6),
            "learning_rate": current_lr,
        }
        epoch_logs.append(log_entry)

        print(
            f"[{args.run_id}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"grad_norm={avg_grad_norm:.4f} lr={current_lr:.6f}"
        )

    _, test_acc = evaluate(model, test_loader, device)
    print(f"[{args.run_id}] final test_acc={test_acc:.4f}")

    save_run(args, epoch_logs, test_acc)


    
def save_run(args, epoch_logs, test_acc):
    os.makedirs(args.output_dir, exist_ok=True)

    run_record = {
        "run_id": args.run_id,
        "label": args.label,
        "injected_problem_type": args.injected_problem_type,
        "config": vars(args),
        "test_acc": round(test_acc, 6),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epoch_logs,
    }

    out_path = os.path.join(args.output_dir, f"{args.run_id}.json")
    with open(out_path, "w") as f:
        json.dump(run_record, f, indent=2)

    print(f"[{args.run_id}] saved run log to {out_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Baseline Fashion-MNIST training run with structured logging.")

        # Identity / labeling (ground truth for evaluation later)
    p.add_argument("--run_id", type=str, required=True)
    p.add_argument("--label", type=str, default="healthy", choices=["healthy", "problem"])
    p.add_argument("--injected_problem_type", type=str, default="none")

        # Core hyperparameters
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

        # Failure-injection knobs (used starting Phase 1.2, default = healthy)
    p.add_argument("--train_subset_frac", type=float, default=1.0, help="Shrink training set (overfitting injection)")
    p.add_argument("--label_noise_frac", type=float, default=0.0, help="Fraction of labels to randomly flip")
    p.add_argument("--freeze_conv1", action="store_true", help="Freeze conv1 (misconfigured-layer injection)")
    p.add_argument("--freeze_conv2", action="store_true", help="Freeze conv2 (misconfigured-layer injection)")
    p.add_argument("--init_scheme", type=str, default="default", choices=["default", "bad"], help="Bad init for vanishing-gradient injection")
    p.add_argument("--output_dir", type=str, default="../data_generation/raw")
    p.add_argument("--dropout_rate", type=float, default=0.25)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)