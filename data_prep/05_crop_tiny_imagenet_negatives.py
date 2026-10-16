"""Generate 24x24 non-face crops from Tiny ImageNet.

Converted mechanically from a notebook cell; not re-run after conversion. Run from the repository root.
"""

import os
import cv2
import numpy as np
import random
from tiny_imagenet_torch import TinyImageNet

# Carpeta de salida
output_folder = "dataset/non_faces"
os.makedirs(output_folder, exist_ok=True)

# Número total de imágenes que queremos generar
TOTAL_IMAGES = 20000  # <- Cambia esto al número que necesites
CROPS_PER_IMAGE = 5
CROP_SIZE = 24

# Detector de caras Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Cargar Tiny-ImageNet
dataset = TinyImageNet(root='./data', train=True, download=False, transform=None)

# Función para generar crops válidos sin caras
def generate_crops(img, num_crops=CROPS_PER_IMAGE, crop_size=CROP_SIZE):
    crops = []
    h, w = img.shape[:2]
    for _ in range(num_crops):
        if w < crop_size or h < crop_size:
            continue
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        crop = img[y:y+crop_size, x:x+crop_size]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_crop, scaleFactor=1.1, minNeighbors=3)
        if len(faces) == 0:
            crops.append(gray_crop)
    return crops

# Generar crops hasta alcanzar TOTAL_IMAGES
saved_count = 0
for idx in range(len(dataset)):
    if saved_count >= TOTAL_IMAGES:
        break
    img, label = dataset[idx]
    img_np = np.array(img)
    crops = generate_crops(img_np, CROPS_PER_IMAGE)
    for i, crop in enumerate(crops):
        if saved_count >= TOTAL_IMAGES:
            break
        out_path = os.path.join(output_folder, f"nonface_{saved_count}.png")
        cv2.imwrite(out_path, crop)
        saved_count += 1
    if idx % 1000 == 0:
        print(f"{idx} imágenes procesadas, {saved_count} crops guardados")

print(f"Listo! {saved_count} imágenes guardadas en {output_folder}")
