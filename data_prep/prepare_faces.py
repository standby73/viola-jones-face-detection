"""Detect, crop (24x24, 10% margin) and split face images into train/test.

Converted mechanically from a notebook cell; not re-run after conversion. Run from the repository root.
"""

import cv2
import os
import random
import numpy as np

# Configuración
input_dir = "face_images"
output_dir = "dataset/faces_clean"
test_dir = "dataset/faces_test"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

TARGET = 30000
SIZE = (24, 24)
MARGIN = 0.1  # 10% margen

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

images = os.listdir(input_dir)
random.seed(42)
random.shuffle(images)

saved = 0
processed = 0

for img_name in images:
    if saved >= TARGET:
        break

    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        if saved >= TARGET:
            break

        dw = int(w * MARGIN)
        dh = int(h * MARGIN)

        x1 = max(0, x - dw)
        y1 = max(0, y - dh)
        x2 = min(width, x + w + dw)
        y2 = min(height, y + h + dh)

        face_roi = gray[y1:y2, x1:x2]

        # ignorar regiones demasiado pequeñas
        if face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
            continue

        # 5. REDIMENSIÓN FINAL
        face_final = cv2.resize(face_roi, SIZE, interpolation=cv2.INTER_AREA).astype(np.uint8)

        if saved < 20_000:  
            out_path = os.path.join(output_dir, f"face_{saved:05d}.png")
            cv2.imwrite(out_path, face_final)
        else:
            out_path = os.path.join(test_dir, f"face_{saved:05d}.png")
            cv2.imwrite(out_path, face_final)
        

        saved += 1

    processed += 1
    if processed % 50 == 0:
        print(f"Procesadas: {processed} | Guardadas: {saved}")

print("\n--- FINALIZADO ---")
print(f"Total caras extraídas: {saved}")
