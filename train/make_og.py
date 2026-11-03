"""Render static/og.png (1200x630 social preview) from the design tokens.

    python train/make_og.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "static" / "og.png"
W, H = 1200, 630
BG, TEXT, MUTED, ACCENT, SURFACE = "#ffffff", "#111111", "#666666", "#2962ff", "#f4f4f5"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    x, y, s = 120, 170, 72
    d.rounded_rectangle((x, y, x + s, y + s), radius=16, outline=TEXT, width=7)
    pts = [(x + 8, y + 56), (x + 26, y + 36), (x + 40, y + 50), (x + 48, y + 42), (x + 64, y + 60)]
    d.line(pts, fill=TEXT, width=7, joint="curve")
    d.ellipse((x + 44, y + 12, x + 60, y + 28), fill=ACCENT)

    d.text((x + s + 28, y - 10), "ML Image Classifier", font=font(78, bold=True), fill=TEXT)
    d.text((x, y + 110), "Find out which animal you are", font=font(44), fill=MUTED)
    d.text((x, y + 180), "Upload a selfie · 10 animals · instant, free", font=font(30), fill=MUTED)

    animals = "🦁 🐯 🐘 🐶 🐱 🐰 🐼 🐻 🐊 🦌"
    d.rounded_rectangle((x, 450, W - x, 560), radius=16, fill=SURFACE)
    try:
        emoji = ImageFont.truetype("seguiemj.ttf", 52)
        d.text((W / 2, 505), animals, font=emoji, fill=TEXT, anchor="mm", embedded_color=True)
    except OSError:
        d.text((W / 2, 505), "lion · tiger · elephant · dog · cat · rabbit · panda · bear · crocodile · deer",
               font=font(28), fill=MUTED, anchor="mm")

    d.rectangle((0, H - 8, W, H), fill=ACCENT)
    im.save(OUT, "PNG", optimize=True)
    print(f"saved {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
