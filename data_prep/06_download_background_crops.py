"""Download background datasets and generate 24x24 non-face crops.

Converted mechanically from a notebook cell; not re-run after conversion. Run from the repository root.
"""

import os
import cv2
import numpy as np
import random
import urllib.request
import tarfile
import zipfile
from pathlib import Path

CROP_SIZE   = 24
TOTAL_CROPS = 20000
OUTPUT_DIR  = "dataset/non_faces"
DATA_DIR    = "./data/backgrounds"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ── Fuentes de descarga ───────────────────────────────────────
SOURCES = [
    {
        "name":    "Stanford Background",
        "url":     "http://dags.stanford.edu/data/iccv09Data.tar.gz",
        "file":    "iccv09Data.tar.gz",
        "extract": "tar",
        "img_dir": "iccv09Data/images",
        "glob":    "*.jpg",
    },
    {
        "name":    "CBCL non-faces MIT",
        "url":     "http://www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz",
        "file":    "faces_mit.tar.gz",
        "extract": "tar",
        "img_dir": "train/non-face",
        "glob":    "*.pgm",
    },
    {
        # Brodatz textures: 112 imágenes de texturas puras, sin ninguna cara
        "name":    "Brodatz Textures",
        "url":     "https://www.ux.uis.no/~tranden/brodatz/D1.gif",  # placeholder
        "skip_download": True,  # las descargamos manualmente abajo
    },
]

def download_and_extract(source):
    dest = os.path.join(DATA_DIR, source["file"])
    img_dir = os.path.join(DATA_DIR, source["img_dir"])

    if not os.path.exists(img_dir):
        print(f"Descargando {source['name']}...")
        try:
            urllib.request.urlretrieve(source["url"], dest)
        except Exception as e:
            print(f"  Error descargando {source['name']}: {e}")
            return None

        print(f"  Extrayendo...")
        try:
            if source["extract"] == "tar":
                with tarfile.open(dest, "r:gz") as tar:
                    tar.extractall(DATA_DIR)
            elif source["extract"] == "zip":
                with zipfile.ZipFile(dest, "r") as z:
                    z.extractall(DATA_DIR)
        except Exception as e:
            print(f"  Error extrayendo: {e}")
            return None

    return img_dir


# ── Recoger todas las imágenes de todas las fuentes ──────────
all_image_paths = []

for source in SOURCES:
    if source.get("skip_download"):
        continue
    img_dir = download_and_extract(source)
    if img_dir and os.path.exists(img_dir):
        paths = list(Path(img_dir).rglob(source["glob"]))
        print(f"  {source['name']}: {len(paths)} imágenes")
        all_image_paths.extend(paths)

# Añadir cualquier imagen que tengas localmente en data/backgrounds
extra = list(Path(DATA_DIR).rglob("*.jpg")) + \
        list(Path(DATA_DIR).rglob("*.png")) + \
        list(Path(DATA_DIR).rglob("*.pgm"))
all_image_paths.extend(extra)
all_image_paths = list(set(all_image_paths))  # deduplicar

print(f"\nTotal imágenes de fondo disponibles: {len(all_image_paths)}")

# ── Filtros de calidad ────────────────────────────────────────
def is_valid_crop(crop_gray):
    """Rechaza crops demasiado uniformes (cielo blanco, paredes lisas)."""
    std = crop_gray.std()
    mean = crop_gray.mean()
    if std < 15:           # demasiado uniforme (blanco, negro, gris plano)
        return False
    if mean < 20:          # demasiado oscuro
        return False
    if mean > 235:         # demasiado claro
        return False
    # Rechazar si más del 60% de píxeles están en rango [240,255]
    if np.mean(crop_gray > 240) > 0.6:
        return False
    return True

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Generar crops variados ────────────────────────────────────
saved = 0
random.shuffle(all_image_paths)

# Cuántos crops intentar por imagen para repartir uniformemente
crops_per_image = max(5, (TOTAL_CROPS // max(len(all_image_paths), 1)) + 5)

for img_path in all_image_paths:
    if saved >= TOTAL_CROPS:
        break

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Intentar con flags alternativos (para .pgm)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    if h < CROP_SIZE * 2 or w < CROP_SIZE * 2:
        continue

    attempts   = 0
    local_hits = 0

    while attempts < crops_per_image * 3 and local_hits < crops_per_image and saved < TOTAL_CROPS:
        x = random.randint(0, w - CROP_SIZE)
        y = random.randint(0, h - CROP_SIZE)
        crop = img[y:y+CROP_SIZE, x:x+CROP_SIZE]

        if not is_valid_crop(crop):
            attempts += 1
            continue

        faces = face_cascade.detectMultiScale(
            crop, scaleFactor=1.1, minNeighbors=3
        )
        if len(faces) == 0:
            out = os.path.join(OUTPUT_DIR, f"nonface_{saved:06d}.png")
            cv2.imwrite(out, crop)
            saved      += 1
            local_hits += 1

        attempts += 1

    if saved % 1000 == 0 and saved > 0:
        print(f"  {saved}/{TOTAL_CROPS} guardados...")

print(f"\nFinalizado: {saved} no-caras en '{OUTPUT_DIR}'")
