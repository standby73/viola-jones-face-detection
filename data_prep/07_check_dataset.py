"""Sanity check of the face / non-face datasets (sizes, counts).

Converted mechanically from a notebook cell; not re-run after conversion. Run from the repository root.
"""

import os
import cv2
import numpy as np

PATH_FACES    = 'dataset/faces'
PATH_NONFACES = 'dataset/non_faces'

# Caras
face_files = [f for f in os.listdir(PATH_FACES) if not f.startswith('.')]
sizes_faces = []
for f in face_files[:20]:
    img = cv2.imread(os.path.join(PATH_FACES, f), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        sizes_faces.append(img.shape)

# No-caras
nonface_files = [f for f in os.listdir(PATH_NONFACES) if not f.startswith('.')]
sizes_nonfaces = []
for f in nonface_files[:20]:
    img = cv2.imread(os.path.join(PATH_NONFACES, f), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        sizes_nonfaces.append(img.shape)

print(f"Total caras:    {len(face_files)}")
print(f"Total no-caras: {len(nonface_files)}")
print(f"Tamaños caras (primeras 20):    {set(sizes_faces)}")
print(f"Tamaños no-caras (primeras 20): {set(sizes_nonfaces)}")

# Muestra estadísticas de píxeles
sample_face = cv2.imread(os.path.join(PATH_FACES, face_files[0]), cv2.IMREAD_GRAYSCALE)
sample_nonface = cv2.imread(os.path.join(PATH_NONFACES, nonface_files[0]), cv2.IMREAD_GRAYSCALE)
print(f"\nCara ejemplo    - mean={sample_face.mean():.1f}  std={sample_face.std():.1f}  shape={sample_face.shape}")
print(f"No-cara ejemplo - mean={sample_nonface.mean():.1f}  std={sample_nonface.std():.1f}  shape={sample_nonface.shape}")
