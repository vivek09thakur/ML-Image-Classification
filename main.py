"""FastAPI app: upload an animal photo, get one of 10 animals back.

GET  /             upload form
POST /predict      form post, renders the result in the same page
POST /api/predict  multipart image -> JSON predictions
"""

import base64
import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError

from Classifier import CLASSIFIER
from Classifier.ImageClassifier import UNSURE_SIMILARITY

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
DECODE_SIDE = 512
THUMB_SIDE = 240
OTHER_OPTIONS = 2


class CachedStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


app = FastAPI(title="Animal Classifier")
app.mount("/static", CachedStaticFiles(directory=STATIC_DIR), name="static")

ASSET_VERSIONS = {p.name: str(int(p.stat().st_mtime)) for p in STATIC_DIR.iterdir() if p.is_file()}


def asset_context(request: Request) -> dict:
    def asset(name: str) -> str:
        return f"{request.url_for('static', path=name)}?v={ASSET_VERSIONS.get(name, '0')}"

    return {"asset": asset}


templates = Jinja2Templates(directory=ROOT / "templates", context_processors=[asset_context])

classifier = CLASSIFIER()
ANIMALS = classifier.labels
ANIMAL_ICONS = {
    "lion": "\U0001F981",
    "tiger": "\U0001F42F",
    "elephant": "\U0001F418",
    "dog": "\U0001F436",
    "cat": "\U0001F431",
    "rabbit": "\U0001F430",
    "panda": "\U0001F43C",
    "bear": "\U0001F43B",
    "crocodile": "\U0001F40A",
    "deer": "\U0001F98C",
}


class UploadError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


async def read_image(photo: UploadFile | None) -> Image.Image:
    if photo is None or not photo.filename:
        raise UploadError("Choose a photo first.")
    data = await photo.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("That photo is over 4 MB. Pick a smaller one or let the page shrink it.", 413)
    try:
        im = Image.open(io.BytesIO(data))
        im.draft("RGB", (DECODE_SIDE, DECODE_SIDE))
        im.load()
    except (UnidentifiedImageError, OSError):
        raise UploadError("That file isn't an image we can read. Use a JPG, PNG or WebP.", 415)
    return ImageOps.exif_transpose(im).convert("RGB")


def classify(im: Image.Image) -> dict:
    predictions = classifier.predict(im)
    for p in predictions:
        p["icon"] = ANIMAL_ICONS.get(p["label"], "")
        p["percent"] = round(p["confidence"] * 100)
    top = predictions[0]
    return {
        "label": top["label"],
        "icon": top["icon"],
        "confidence": top["confidence"],
        "percent": top["percent"],
        "unsure": top["similarity"] < UNSURE_SIMILARITY,
        "others": predictions[1 : 1 + OTHER_OPTIONS],
        "predictions": predictions,
    }


def thumbnail_data_url(im: Image.Image) -> str:
    thumb = ImageOps.fit(im, (THUMB_SIDE, THUMB_SIDE))
    buf = io.BytesIO()
    thumb.save(buf, "JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render(request: Request, status: int = 200, **context) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", context, status_code=status)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render(request)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, photo: UploadFile | None = File(None)):
    try:
        im = await read_image(photo)
    except UploadError as e:
        return render(request, status=e.status, error=e.message)
    return render(request, result=classify(im), thumb=thumbnail_data_url(im))


@app.post("/api/predict")
async def api_predict(photo: UploadFile | None = File(None)):
    try:
        im = await read_image(photo)
    except UploadError as e:
        raise HTTPException(e.status, e.message)
    return JSONResponse(classify(im))


@app.get("/health")
async def health():
    return {"ok": True, "animals": ANIMALS}
