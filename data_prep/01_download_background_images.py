"""Download random background images (loremflickr).

Converted mechanically from a notebook cell; not re-run after conversion. Run from the repository root.
"""

import os
import cv2
import requests
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

OUTPUT_DIR = "raw_step1_download"
TOTAL_IMAGES = 30000

os.makedirs(OUTPUT_DIR, exist_ok=True)

tags = ["nature", "architecture", "landscape", "city", "texture", "objects"]


def download(i):
    tag = random.choice(tags)
    url = f"https://loremflickr.com/1024/768/{tag}"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return

        img_array = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return

        cv2.imwrite(f"{OUTPUT_DIR}/img_{i:05d}.jpg", img)

        if i % 500 == 0:
            print(f"[OK] {i}")

    except:
        pass


print("Descargando imágenes...")

with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(download, range(TOTAL_IMAGES))

print("✔ Descarga completada")
