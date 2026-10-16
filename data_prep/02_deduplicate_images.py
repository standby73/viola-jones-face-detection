"""Remove near-duplicate images with perceptual hashing.

Converted mechanically from a notebook cell; not re-run after conversion. Run from the repository root.
"""

import os
import imagehash
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

INPUT_DIR = "raw_step1_download"
OUTPUT_DIR = "raw_step2_dedup"

os.makedirs(OUTPUT_DIR, exist_ok=True)

HASH_SIZE = 8

hash_dict = {}
to_delete = set()


def compute_hash(path):
    try:
        img = Image.open(path).convert("L")
        h = imagehash.phash(img, hash_size=HASH_SIZE)
        return path, h
    except:
        return path, None


images = [
    os.path.join(INPUT_DIR, f)
    for f in os.listdir(INPUT_DIR)
]

print(f"Total imágenes: {len(images)}")

results = []

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(compute_hash, img) for img in images]

    for f in tqdm(as_completed(futures), total=len(futures)):
        results.append(f.result())


# detectar duplicados
for path, h in results:
    if h is None:
        continue

    duplicate = False

    for existing in hash_dict:
        if h - existing <= 5:
            duplicate = True
            break

    if not duplicate:
        hash_dict[h] = path
        os.rename(path, os.path.join(OUTPUT_DIR, os.path.basename(path)))
    else:
        try:
            os.remove(path)
        except:
            pass

print("✔ Duplicados eliminados")
