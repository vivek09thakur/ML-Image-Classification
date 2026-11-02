"""Collect an open-licensed image dataset for the 10 animal classes.

Images are fetched with Bing image search restricted to licences that allow
reuse and modification, then verified, converted to RGB JPEG, downscaled and
de-duplicated into data/raw/<class>/.

    python train/collect_dataset.py --per-class 10
"""

import argparse
import hashlib
import logging
import shutil
from pathlib import Path

import truststore
from icrawler.builtin import BingImageCrawler
from PIL import Image

truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TMP_DIR = ROOT / "data" / "tmp"

CLASS_QUERIES = {
    "lion": ["lion animal", "african lion wildlife", "lioness photo"],
    "tiger": ["bengal tiger", "tiger wildlife photo", "siberian tiger"],
    "elephant": ["elephant animal", "african elephant photo", "asian elephant"],
    "dog": ["dog pet photo", "domestic dog outdoors", "puppy dog"],
    "cat": ["domestic cat photo", "house cat pet", "kitten cat"],
    "rabbit": ["rabbit animal", "pet bunny rabbit", "wild rabbit photo"],
    "panda": ["giant panda", "panda bear bamboo", "giant panda photo"],
    "bear": ["brown bear wildlife", "grizzly bear photo", "black bear animal"],
    "crocodile": ["crocodile animal", "crocodile river photo", "saltwater crocodile"],
    "deer": ["deer animal", "deer in forest photo", "white tailed deer"],
}

MAX_SIDE = 512
MIN_SIDE = 96


def download(class_name: str, query: str, max_num: int) -> Path:
    out = TMP_DIR / class_name
    out.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(
        storage={"root_dir": str(out)},
        downloader_threads=4,
        log_level=logging.WARNING,
    )
    crawler.crawl(
        keyword=query,
        max_num=max_num,
        filters={"license": "commercial,modify", "type": "photo"},
        file_idx_offset="auto",
    )
    return out


def clean(src: Path, class_name: str, limit: int) -> int:
    dst = RAW_DIR / class_name
    dst.mkdir(parents=True, exist_ok=True)
    seen = {p.stem for p in dst.glob("*.jpg")}
    kept = 0
    for path in sorted(src.iterdir()):
        if len(seen) >= limit:
            break
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im = im.convert("RGB")
                if min(im.size) < MIN_SIDE:
                    continue
                im.thumbnail((MAX_SIDE, MAX_SIDE))
                digest = hashlib.md5(im.tobytes()).hexdigest()[:16]
                if digest in seen:
                    continue
                im.save(dst / f"{digest}.jpg", "JPEG", quality=88)
                seen.add(digest)
                kept += 1
        except Exception:
            continue
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--classes", nargs="*", default=list(CLASS_QUERIES))
    args = parser.parse_args()

    for class_name in args.classes:
        have = len(list((RAW_DIR / class_name).glob("*.jpg")))
        for query in CLASS_QUERIES[class_name]:
            if have >= args.per_class:
                break
            print(f"[{class_name}] downloading '{query}'...")
            tmp = download(class_name, query, (args.per_class - have) * 2)
            have += clean(tmp, class_name, args.per_class)
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"[{class_name}] {have} images")

    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
