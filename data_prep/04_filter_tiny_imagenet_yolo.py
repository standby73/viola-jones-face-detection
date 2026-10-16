"""Filter Tiny ImageNet validation images with YOLO.

Converted mechanically from a notebook cell; not re-run after conversion. Run from the repository root.
"""

import os
import shutil
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
from tqdm import tqdm
import threading

# --- PATHS ---
DATASET_ROOT = "data/tiny-imagenet-200"
OUTPUT_DIR = "raw_negatives_clean"

# TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val", "images")
TEST_DIR = os.path.join(DATASET_ROOT, "test", "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CONFIG ---
MAX_IMAGES = 10000

# --- YOLO ---
model = YOLO("yolov8l.pt")
PERSON_CLASS = 0

# --- CONTROL THREAD-SAFE ---
lock = threading.Lock()
saved_count = 0


def has_person(img):
    results = model(img, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == PERSON_CLASS and conf > 0.3:
            return True

    return False


def collect_images():
    images = []

    # # TRAIN
    # for cls_folder in os.listdir(TRAIN_DIR):
    #     img_dir = os.path.join(TRAIN_DIR, cls_folder, "images")
    #     if os.path.exists(img_dir):
    #         for f in os.listdir(img_dir):
    #             if f.endswith(".JPEG"):
    #                 images.append(os.path.join(img_dir, f))

    # VAL
    for f in os.listdir(VAL_DIR):
        if f.endswith(".JPEG"):
            images.append(os.path.join(VAL_DIR, f))

    # TEST
    for f in os.listdir(TEST_DIR):
        if f.endswith(".JPEG"):
            images.append(os.path.join(TEST_DIR, f))

    return images


images = collect_images()
print(f"Total imágenes disponibles: {len(images)}")


def process_image(path):
    global saved_count

    # parar si ya llegamos al límite
    with lock:
        if saved_count >= MAX_IMAGES:
            return 0

    img = cv2.imread(path)
    if img is None:
        return 0

    if has_person(img):
        return 0

    # copiar imagen
    filename = os.path.basename(path)
    dst = os.path.join(OUTPUT_DIR, filename)

    shutil.copy2(path, dst)

    with lock:
        if saved_count < MAX_IMAGES:
            saved_count += 1
            return 1
        else:
            return 0


print("\nFiltrando imágenes...")

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(process_image, img) for img in images]

    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass

print("\n--- FINAL ---")
print(f"Imágenes guardadas: {saved_count}")
