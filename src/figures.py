"""Regenerate report charts from dataset, TensorBoard events and test results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.data import CLASSES, DATA, ROOT

OUT = ROOT / "results"
RUNS = ROOT / "runs"
ABBR = ("AC", "FO", "HV", "HW", "IN", "PA", "PC", "RE", "RI", "SL")
COLORS = ("#52798e", "#ba7438", "#35866b")


def charts():
    audit = json.loads((DATA / "audit.json").read_text())
    metrics = json.loads((OUT / "test_metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(8.2, 2.7), layout="constrained")
    baseline = np.zeros(len(CLASSES))
    for color, split in zip(COLORS, ("train", "val", "test"), strict=True):
        counts = [audit["counts"][label][split] for label in CLASSES]
        ax.bar(ABBR, counts, bottom=baseline, color=color, label=split)
        baseline += counts
    ax.legend(frameon=False, ncol=3)
    ax.set(ylabel="Images", xlabel="Class")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "data_distribution.png", dpi=150)
    plt.close(fig)

    for kind in ("Loss", "Accuracy"):
        fig, axs = plt.subplots(1, 2, figsize=(8.3, 3.3), sharey=True, layout="constrained")
        colors = iter(COLORS)
        for run in sorted(RUNS.iterdir()):
            accumulator = EventAccumulator(str(run)).Reload()
            color = next(colors)
            for axis, split in zip(axs, ("train", "val"), strict=True):
                events = accumulator.Scalars(f"{kind}/{split}")
                axis.plot([entry.step for entry in events], [entry.value for entry in events],
                          color=color, marker="o", markersize=3, label=run.name)
                axis.set(title=split.capitalize(), xlabel="Epoch", xticks=range(1, 9))
                axis.grid(alpha=.2)
                axis.spines[["top", "right"]].set_visible(False)
        axs[0].set_ylabel(kind)
        axs[1].legend(frameon=False, fontsize=8)
        fig.savefig(OUT / f"{kind.lower()}_curves.png", dpi=150)
        plt.close(fig)

    matrix = np.array(metrics["confusion"])
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    colored = ax.imshow(matrix, cmap="Greens")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center", fontsize=8,
                color="white" if value > matrix.max() / 2 else "#17362b")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=ABBR, yticklabels=ABBR,
           xlabel="Predicted class", ylabel="True class")
    fig.colorbar(colored, ax=ax, label="Images")
    fig.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def thumbnails(paths, names, columns, destination):
    width, height, cell = 144, 117, 100
    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new("RGB", (width * columns, height * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, name) in enumerate(zip(paths, names, strict=True)):
        x, y = index % columns * width, index // columns * height
        with Image.open(path) as image:
            canvas.paste(image.resize((cell, cell), Image.Resampling.LANCZOS),
                         (x + (width - cell) // 2, y))
        draw.text((x + width // 2, y + cell + 4), name, fill="#153a34", anchor="mt")
    canvas.save(destination)


def examples():
    samples = [next((DATA / "samples" / label).glob("*.jpg")) for label in CLASSES]
    thumbnails(samples, ABBR, 5, OUT / "dataset_samples.png")
    with (OUT / "test_predictions.csv").open() as source:
        predictions = list(csv.DictReader(source))
    correct = [next(row for row in predictions
                    if row["actual"] == row["predicted"] == label) for label in list(CLASSES)[:5]]
    wrong = [next(row for row in predictions if row["actual"] == label
                  and row["actual"] != row["predicted"]) for label in ("Highway", "PermanentCrop")]
    examples = correct + wrong
    names = [f"{i + 1}  {ABBR[list(CLASSES).index(row['actual'])]} > "
             f"{ABBR[list(CLASSES).index(row['predicted'])]}" for i, row in enumerate(examples)]
    thumbnails([DATA / row["path"] for row in examples], names, 4, OUT / "test_cases.png")
    (OUT / "test_cases.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")


if __name__ == "__main__":
    charts()
    examples()
