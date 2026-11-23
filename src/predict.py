"""Load the saved CNN and classify a single RGB image."""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from src.data import CLASSES, image_transform
from src.model import CHECKPOINT, load_model

PREPROCESS = image_transform()


@torch.inference_mode()
def predict_image(model, image):
    started = time.perf_counter()
    pixels = PREPROCESS(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
    values, indices = model(pixels).softmax(dim=1).topk(3, dim=1)
    labels = list(CLASSES)
    predictions = [
        {"id": labels[index], "label": CLASSES[labels[index]], "probability": value}
        for value, index in zip(values[0].tolist(), indices[0].tolist(), strict=True)
    ]
    return {"top3": predictions, "inference_ms": round((time.perf_counter() - started) * 1000, 2)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    network, metadata = load_model(args.model)
    with Image.open(args.image) as source:
        print(json.dumps(predict_image(network, source), ensure_ascii=False, indent=2))
