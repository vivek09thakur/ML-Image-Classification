"""Build the animal classifier from data/raw/<class>/*.jpg.

Same idea as the original project (ImageNet features + cosine similarity),
but lightweight: a MobileNetV3 .tflite embedder replaces VGG16, and each class
is summarised by one mean feature vector ("prototype"). No weight training is
needed, so 10 images per class is enough.

Outputs to model/:
    embedder.tflite   pretrained MediaPipe image embedder (downloaded once)
    prototypes.npz    L2-normalised class prototypes + labels

    python train/build_model.py [--variant small|large]   (default: large, 10 MB)
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Classifier import CLASSIFIER  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "model"

EMBEDDER_URLS = {
    "small": "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite",
    "large": "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/1/mobilenet_v3_large.tflite",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=EMBEDDER_URLS, default="large")
    args = parser.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    embedder_path = MODEL_DIR / "embedder.tflite"
    if not embedder_path.exists():
        print(f"downloading {args.variant} embedder...")
        urllib.request.urlretrieve(EMBEDDER_URLS[args.variant], embedder_path)
    classifier = CLASSIFIER(embedder_path, prototypes_path=None)

    labels = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())
    features: dict[str, list[np.ndarray]] = {}
    for label in labels:
        paths = sorted((RAW_DIR / label).glob("*.jpg"))
        features[label] = [classifier.extract_features(Image.open(p)) for p in paths]
        print(f"[{label}] embedded {len(paths)} images")

    correct = total = 0
    for label in labels:
        for i, vec in enumerate(features[label]):
            protos = []
            for other in labels:
                vecs = [v for j, v in enumerate(features[other]) if not (other == label and j == i)]
                protos.append(np.mean(vecs, axis=0))
            protos = np.stack(protos)
            protos /= np.linalg.norm(protos, axis=1, keepdims=True)
            correct += labels[int(np.argmax(protos @ vec))] == label
            total += 1
    print(f"leave-one-out accuracy: {correct}/{total} = {correct / total:.1%}")

    prototypes = np.stack([np.mean(features[l], axis=0) for l in labels])
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    np.savez(MODEL_DIR / "prototypes.npz", prototypes=prototypes.astype(np.float32), labels=np.array(labels))
    (MODEL_DIR / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"saved {MODEL_DIR / 'prototypes.npz'} ({embedder_path.stat().st_size / 1e6:.1f} MB embedder)")


if __name__ == "__main__":
    main()
