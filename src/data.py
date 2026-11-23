"""Verify the EuroSAT RGB release and create reproducible grouped splits."""
import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
MANIFEST = DATA / "dataset_manifest.csv"
URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
ARCHIVE_SHA = "b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"
CLASSES = {
    "AnnualCrop": "一年生作物", "Forest": "森林", "HerbaceousVegetation": "草本植被",
    "Highway": "高速公路", "Industrial": "工业区", "Pasture": "牧场",
    "PermanentCrop": "多年生作物", "Residential": "住宅区", "River": "河流",
    "SeaLake": "海洋与湖泊",
}
SPLITS = ("train", "val", "test", "demo")
SEED = 2026
SIZE = 32
MEAN = (0.344, 0.380, 0.408)
STD = (0.203, 0.137, 0.115)


def image_transform():
    return transforms.Compose([
        transforms.Resize((SIZE, SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def prepare(archive):
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(".part")
        print(f"Downloading {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=90) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(archive)
    with archive.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != ARCHIVE_SHA:
            raise ValueError("EuroSAT RGB ZIP SHA-256 mismatch. Check the source archive.")
    groups, excluded = {}, []
    official = Counter()
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as source:
        for member in sorted(source.namelist()):
            parts = member.split("/")
            if len(parts) != 3 or parts[0] != "EuroSAT_RGB" or parts[1] not in CLASSES:
                continue
            label, filename = parts[1:]
            if not filename.endswith(".jpg"):
                continue
            official[label] += 1
            payload = source.read(member)
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    if image.size != (64, 64) or image.mode != "RGB":
                        raise ValueError("unexpected image size or mode")
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
            except (OSError, ValueError) as error:
                excluded.append({"source": member, "reason": str(error)})
                continue
            relative = Path(label) / filename
            groups.setdefault(digest, []).append((label, relative))
            destination = raw / relative
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(payload)
    if sum(official.values()) != 27000 or set(official) != set(CLASSES):
        raise ValueError(f"Unexpected dataset contents: {official}")
    unique = {label: [] for label in CLASSES}
    for digest, matches in sorted(groups.items()):
        label, relative = matches[0]
        conflict = len({item[0] for item in matches}) > 1
        rejected = matches if conflict else matches[1:]
        for _, path in rejected:
            excluded.append({
                "source": path.as_posix(),
                "reason": "label conflict" if conflict else "duplicate decoded RGB",
                "kept": "" if conflict else relative.as_posix(),
            })
        if not conflict:
            unique[label].append((relative, digest))
    rows = []
    for label, entries in unique.items():
        random.Random(f"{SEED}:{label}").shuffle(entries)
        usable = len(entries) - 3
        train_end, val_end = int(usable * 0.70), int(usable * 0.85)
        partitions = [entries[:train_end], entries[train_end:val_end],
                      entries[val_end:usable], entries[usable:]]
        for split, items in zip(SPLITS, partitions, strict=True):
            for relative, digest in items:
                rows.append({
                    "split": split, "label": label,
                    "path": (Path("raw") / relative).as_posix(), "pixel_sha256": digest,
                })
            if split in ("test", "demo"):
                for relative, _ in items[:1 if split == "test" else 3]:
                    dest = DATA / ("samples" if split == "test" else "demo") / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(raw / relative, dest)
    with MANIFEST.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["split", "label", "path", "pixel_sha256"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "seed": SEED, "archive_sha256": ARCHIVE_SHA, "original_counts": dict(official),
        "counts": {label: dict(Counter(row["split"] for row in rows if row["label"] == label))
                   for label in CLASSES},
        "excluded": excluded,
    }
    (DATA / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print({split: sum(row["split"] == split for row in rows) for split in SPLITS}, flush=True)


class EuroSATSplit(Dataset):
    def __init__(self, split):
        with MANIFEST.open(newline="", encoding="utf-8") as stream:
            self.rows = [row for row in csv.DictReader(stream) if row["split"] == split]
        self.label_ids = {label: index for index, label in enumerate(CLASSES)}
        self.transform = image_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(DATA / row["path"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, self.label_ids[row["label"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DATA / "EuroSAT_RGB.zip")
    prepare(parser.parse_args().archive)
