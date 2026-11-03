"""Smoke tests for the web app. Run: python -m pytest -q"""

import glob
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import app  # noqa: E402

client = TestClient(app)
SAMPLES = {Path(p).parent.name: p for p in glob.glob("data/raw/*/*.jpg")}


def test_index_renders_form():
    r = client.get("/")
    assert r.status_code == 200
    assert 'name="photo"' in r.text
    assert "Choose a photo" in r.text


def test_predict_without_file_shows_error():
    r = client.post("/predict", data={})
    assert r.status_code == 400
    assert "Choose a photo first" in r.text


def test_predict_rejects_non_image():
    r = client.post("/api/predict", files={"photo": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_api_predict_returns_ranked_classes():
    im = Image.new("RGB", (64, 64), "green")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    r = client.post("/api/predict", files={"photo": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == body["predictions"][0]["label"]
    assert len(body["predictions"]) == 10


@pytest.mark.skipif(not SAMPLES, reason="dataset not present")
@pytest.mark.parametrize("label", sorted(SAMPLES))
def test_predict_sample_image(label):
    with open(SAMPLES[label], "rb") as f:
        r = client.post("/predict", files={"photo": (f"{label}.jpg", f.read(), "image/jpeg")})
    assert r.status_code == 200
    assert "Try another photo" in r.text
