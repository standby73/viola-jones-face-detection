"""Filter downloaded backgrounds with YOLO (keeps clean background images).

Converted mechanically from a notebook cell; not re-run after conversion. Run from the repository root.
"""

import os
import cv2
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
from tqdm import tqdm
import threading

INPUT_DIR = "raw_step2_dedup"
OUTPUT_DIR = "background_clean"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_IMAGES = 20000

model = YOLO("yolov8n.pt")
PERSON_CLASS = 0

lock = threading.Lock()
saved = 0


def has_person(img):
    results = model(img, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == PERSON_CLASS and conf > 0.3:
            return True

    return False


def process(path):
    global saved

    with lock:
        if saved >= MAX_IMAGES:
            return

    img = cv2.imread(path)
    if img is None:
        return

    if has_person(img):
        return

    with lock:
        if saved >= MAX_IMAGES:
            return
        saved += 1

    shutil.copy2(path, os.path.join(OUTPUT_DIR, os.path.basename(path)))


images = [
    os.path.join(INPUT_DIR, f)
    for f in os.listdir(INPUT_DIR)
]

print(f"Imágenes a procesar: {len(images)}")

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(process, img) for img in images]

    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass

print(f"✔ Finalizado. Imágenes limpias: {saved}")
