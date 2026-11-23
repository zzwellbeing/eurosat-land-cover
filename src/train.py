"""Train one baseline or all three planned learning-rate experiments."""
import argparse
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.data import CLASSES, ROOT, SEED, EuroSATSplit
from src.model import CHECKPOINT, MODEL_NAME, LandCoverCNN

EXPERIMENTS = {"cnn_lr0003": 0.0003, "cnn_lr001": 0.001, "cnn_lr003": 0.003}


def cached_split(split):
    """Run the deterministic transforms once for each sample before the first epoch."""
    loader = DataLoader(EuroSATSplit(split), batch_size=256, shuffle=False)
    batches = list(loader)
    return TensorDataset(
        torch.cat([pixels for pixels, _ in batches]),
        torch.cat([labels for _, labels in batches]),
    )


def pass_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for pixels, labels in loader:
            pixels, labels = pixels.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(pixels)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def save_checkpoint(path, checkpoint):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(name, args, train_data, val_data, device):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    model = LandCoverCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=EXPERIMENTS[name], weight_decay=0.0001)
    run = args.runs / name
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"{run} already has results. Use a fresh --runs directory.")
    history, best_key, best = [], (-1, -math.inf), None
    started = time.perf_counter()
    with SummaryWriter(str(run)) as writer:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = pass_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc = pass_epoch(model, val_loader, criterion, device)
            metrics = {
                "Loss/train": train_loss, "Loss/val": val_loss, "Accuracy/train": train_acc,
                "Accuracy/val": val_acc, "LearningRate": optimizer.param_groups[0]["lr"],
            }
            for tag, value in metrics.items():
                writer.add_scalar(tag, value, epoch)
            writer.flush()
            history.append({"epoch": epoch, **metrics})
            print(f"{name} epoch {epoch:02d} train {train_loss:.4f}/{train_acc:.2%} "
                  f"val {val_loss:.4f}/{val_acc:.2%}", flush=True)
            key = (val_acc, -val_loss)
            if key > best_key:
                best_key = key
                best = {
                    "state_dict": {key: value.detach().cpu().clone()
                                   for key, value in model.state_dict().items()},
                    "labels": list(CLASSES), "name": MODEL_NAME, "experiment": name,
                    "epoch": epoch, "val_accuracy": val_acc, "val_loss": val_loss,
                }
                save_checkpoint(args.output.parent / f"{name}.pth", best)
    metadata = {
        "experiment": name, "seed": SEED, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": EXPERIMENTS[name], "weight_decay": 0.0001, "optimizer": "AdamW",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device), "threads": args.threads, "python": platform.python_version(),
        "torch": str(torch.__version__), "platform": platform.platform(),
        "seconds": round(time.perf_counter() - started, 2), "history": history,
        "best_epoch": best["epoch"], "best_val_accuracy": best["val_accuracy"],
        "best_val_loss": best["val_loss"],
    }
    (run / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["all", *EXPERIMENTS], default="all")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--output", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda":
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print("Preprocessing train and validation images", flush=True)
    train_data, val_data = cached_split("train"), cached_split("val")
    names = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    candidates = [train(name, args, train_data, val_data, device) for name in names]
    best = max(candidates, key=lambda item: (item["val_accuracy"], -item["val_loss"]))
    save_checkpoint(args.output, best)
    print(f"Saved {args.output} from {best['experiment']} epoch {best['epoch']}", flush=True)


if __name__ == "__main__":
    main()
