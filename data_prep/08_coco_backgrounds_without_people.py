"""Select COCO images without people as extra backgrounds.

Converted mechanically from a notebook cell; not re-run after conversion. Run from the repository root.
"""

from datasets import load_dataset
import os

# Ruta donde quieres guardar (cámbiala)
output_dir = "background_clean"

os.makedirs(output_dir, exist_ok=True)

# Cargar dataset
dataset = load_dataset("detection-datasets/coco", split="train")

# Filtrar imágenes SIN personas
def no_person(example):
    return 0 not in example["objects"]["category"]

filtered = dataset.filter(no_person)

# Quedarte con 10k
subset = filtered.select(range(10000))

# Guardar imágenes
for i, example in enumerate(subset):
    img = example["image"]
    img.save(os.path.join(output_dir, f"image_{i}.jpg"))
