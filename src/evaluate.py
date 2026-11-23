"""Evaluate the selected checkpoint once on the independent test split."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import CLASSES, ROOT, EuroSATSplit
from src.model import CHECKPOINT, load_model


@torch.inference_mode()
def evaluate(model_path, output):
    torch.set_num_threads(4)
    model, checkpoint = load_model(model_path)
    dataset = EuroSATSplit("test")
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    labels = list(CLASSES)
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    results, total_loss, top3_hits = [], 0.0, 0
    for pixels, targets in loader:
        logits = model(pixels)
        total_loss += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        probabilities, indices = logits.softmax(dim=1).topk(3, dim=1)
        top3_hits += (indices == targets[:, None]).any(dim=1).sum().item()
        for target, predicted, scores in zip(targets.tolist(), indices.tolist(),
                                             probabilities.tolist(), strict=True):
            row = dataset.rows[len(results)]
            confusion[target, predicted[0]] += 1
            results.append({
                "path": row["path"], "actual": labels[target], "predicted": labels[predicted[0]],
                "confidence": scores[0], "top3": "|".join(labels[i] for i in predicted),
            })
    diagonal = confusion.diagonal()
    support, predicted_counts = confusion.sum(axis=1), confusion.sum(axis=0)
    metrics = {
        "experiment": checkpoint["experiment"], "epoch": checkpoint["epoch"],
        "n": len(dataset), "loss": total_loss / len(dataset),
        "top1": float(diagonal.sum() / len(dataset)), "top3": top3_hits / len(dataset),
        "macro_f1": float((2 * diagonal / np.maximum(support + predicted_counts, 1)).mean()),
        "classes": labels, "confusion": confusion.tolist(),
        "per_class": {label: {"support": int(support[i]), "correct": int(diagonal[i]),
                              "recall": float(diagonal[i] / support[i])}
                      for i, label in enumerate(labels)},
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    encoded = json.dumps(metrics, indent=2)
    (output / "test_metrics.json").write_text(encoded, encoding="utf-8")
    print({key: metrics[key] for key in ("experiment", "epoch", "n", "loss", "top1", "top3", "macro_f1")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    evaluate(args.model, args.output)
