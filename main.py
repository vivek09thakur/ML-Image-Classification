"""FastAPI app: upload an animal photo, get one of 10 animals back.

GET  /             upload form
POST /predict      form post, renders the result in the same page
POST /api/predict  multipart image -> JSON predictions
"""

import base64
import io
import os
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
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

SITE = {
    "name": "ML Image Classifier",
    "tagline": "Find out which animal you are",
    "description": "Upload a selfie and find out which of 10 animals you look like — lion, tiger, elephant, dog, cat, rabbit, panda, bear, crocodile or deer. Free, instant, runs on a tiny on-device ML model.",
    "url": os.environ.get("SITE_URL", "https://ml-image-classification.vercel.app/"),
    "author": "Vivek Thakur",
    "author_url": "https://vishme.vercel.app",
}


class CachedStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


app = FastAPI(title="Animal Classifier")
app.mount("/static", CachedStaticFiles(directory=STATIC_DIR), name="static")


def asset_context(request: Request) -> dict:
    def asset(name: str) -> str:
        path = STATIC_DIR / name
        version = int(path.stat().st_mtime) if path.exists() else 0
        return f"{request.url_for('static', path=name)}?v={version}"

    return {"asset": asset, "year": date.today().year, "site": SITE}


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
    context.setdefault("icons", [ANIMAL_ICONS[a] for a in ANIMALS if a in ANIMAL_ICONS][:5])
    return templates.TemplateResponse(request, "index.html", context, status_code=status)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render(request)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, photo: UploadFile | None = File(None)):
    try:
        im = await read_image(photo)
    except UploadError as e:
        return render(request, status=e.status, error=e.message, noindex=True)
    return render(request, result=classify(im), thumb=thumbnail_data_url(im), noindex=True)


@app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
async def robots():
    return f"User-agent: *\nAllow: /\nDisallow: /predict\nDisallow: /api/\nSitemap: {SITE['url']}sitemap.xml\n"


@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap():
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{SITE['url']}</loc><changefreq>monthly</changefreq><priority>1.0</priority></url>"
        "</urlset>"
    )
    return Response(xml, media_type="application/xml")


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
