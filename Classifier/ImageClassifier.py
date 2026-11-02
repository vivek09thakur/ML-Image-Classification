"""Animal classifier: TFLite image embedder + cosine similarity to class prototypes.

Same approach as the original project (ImageNet features + cosine similarity)
with a 10 MB MobileNetV3 .tflite embedder instead of VGG16, so it runs on
Vercel without TensorFlow.
"""

from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageOps

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
EMBEDDER_PATH = MODEL_DIR / "embedder.tflite"
PROTOTYPES_PATH = MODEL_DIR / "prototypes.npz"

TEMPERATURE = 12.0
UNSURE_SIMILARITY = 0.25


class CLASSIFIER:
    def __init__(self, embedder_path: Path = EMBEDDER_PATH, prototypes_path: Path | None = PROTOTYPES_PATH):
        self.interpreter = Interpreter(model_path=str(embedder_path), num_threads=2)
        self.interpreter.allocate_tensors()
        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]
        self.input_size = int(self._input["shape"][1])

        self.labels: list[str] = []
        self.prototypes = np.zeros((0, 1), dtype=np.float32)
        if prototypes_path is not None and Path(prototypes_path).exists():
            data = np.load(prototypes_path)
            self.labels = [str(l) for l in data["labels"]]
            self.prototypes = data["prototypes"].astype(np.float32)

    def load_and_preprocess_image(self, im: Image.Image) -> np.ndarray:
        size = (self.input_size, self.input_size)
        im = ImageOps.fit(im.convert("RGB"), size, Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)[None] / 255.0

    def extract_features(self, im: Image.Image) -> np.ndarray:
        self.interpreter.set_tensor(self._input["index"], self.load_and_preprocess_image(im))
        self.interpreter.invoke()
        vec = self.interpreter.get_tensor(self._output["index"]).reshape(-1).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    @staticmethod
    def get_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
        return float(np.dot(feature1, feature2))

    def predict(self, im: Image.Image) -> list[dict]:
        feature = self.extract_features(im)
        sims = self.prototypes @ feature
        exp = np.exp((sims - sims.max()) * TEMPERATURE)
        confidences = exp / exp.sum()
        order = np.argsort(-sims)
        return [
            {"label": self.labels[i], "similarity": float(sims[i]), "confidence": float(confidences[i])}
            for i in order
        ]
