"""Generate 24x24 negative crops from held-out background images for testing.

Converted mechanically from a notebook cell; not re-run after conversion. Run from the repository root.
"""

import os
import random
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# =========================
# CONFIGURACIÓN
# =========================
input_dir = "dataset/background_test"
crops_dir = "dataset/background_crops"
os.makedirs(crops_dir, exist_ok=True)

SIZE = 24
CROPS_PER_IMAGE = 200

imagenes = [f for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

random.shuffle(imagenes)

print(f"Imágenes encontradas: {len(imagenes)}")


# =========================
# FUNCIÓN DE PROCESADO
# =========================
def process_image(img_name):

    img_path = os.path.join(input_dir, img_name)
    crops = []

    try:
        with Image.open(img_path) as img:
            img = img.convert("L")  # grayscale uint8
            w, h = img.size

            if w < SIZE or h < SIZE:
                return []

            for _ in range(CROPS_PER_IMAGE):
                x = random.randint(0, w - SIZE)
                y = random.randint(0, h - SIZE)

                crop = img.crop((x, y, x + SIZE, y + SIZE))
                crop_np = np.array(crop, dtype=np.uint8)

                crops.append(crop_np)

    except Exception:
        pass

    return crops


# =========================
# EJECUCIÓN PARALELA
# =========================
all_crops = []

with ProcessPoolExecutor() as executor:
    for i, result in enumerate(executor.map(process_image, imagenes)):
        all_crops.extend(result)

        if i % 50 == 0:
            print(f"Procesadas {i} imágenes")

print(f"Total crops generados: {len(all_crops)}")


def is_valid_crop(crop_gray):
    """Rechaza crops demasiado uniformes (cielo blanco, paredes lisas)."""
    std = crop_gray.std()
    mean = crop_gray.mean()
    if std < 15: return False
    if mean < 20: return False
    if mean > 235: return False
    if np.mean(crop_gray > 240) > 0.6: return False
    return True
# =========================
# GUARDADO
# =========================
saved = 0
for i, crop in enumerate(all_crops):
    if not is_valid_crop(crop):
        continue
    Image.fromarray(crop, mode="L").save(
        os.path.join(crops_dir, f"crop_{i:06d}.png")
    )
    saved+=1

    if saved % 1000 == 0:
        print(f"Guardados {saved} crops")

    if saved >= 100_000:
        break

print("✔ FINALIZADO")
