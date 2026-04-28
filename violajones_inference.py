"""
Viola-Jones Inference - Cámara en Tiempo Real
==============================================
Carga N etapas desde una carpeta de checkpoints y corre detección
sobre la cámara con sliding window + pirámide de escalas + NMS.

Uso:
    python inference_vj.py --stages checkpoints_total --num 12
    python inference_vj.py --stages checkpoints_total --num 12 --scale 1.2 --step 4
    python inference_vj.py --stages checkpoints_total --num 12 --image foto.jpg
"""

import os
import cv2
import pickle
import argparse
import time
import numpy as np
from pathlib import Path
from collections import deque


# ──────────────────────────────────────────────────────────────
# HELPERS DE INTEGRAL IMAGE Y FEATURES (igual que en entrenamiento)
# ──────────────────────────────────────────────────────────────

def integral_image(img):
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.int64), axis=0), axis=1)
    return ii


def normalize_variance(ii, ii2, h, w):
    N    = h * w
    mean = (ii[h, w] - ii[0, w] - ii[h, 0] + ii[0, 0]) / N
    ssq  = ii2[h, w] - ii2[0, w] - ii2[h, 0] + ii2[0, 0]
    var  = (ssq / N) - (mean ** 2)
    return np.sqrt(max(var, 0))


def sum_region(ii, x1, y1, x2, y2):
    return int(ii[y2+1, x2+1] - ii[y1, x2+1] - ii[y2+1, x1] + ii[y1, x1])


def compute_feature(ii, x, y, w, h, ftype=0):
    if ftype == 0:
        hw = w // 2
        return (sum_region(ii, x, y, x+hw-1, y+h-1)
              - sum_region(ii, x+hw, y, x+w-1, y+h-1))
    elif ftype == 1:
        hh = h // 2
        return (sum_region(ii, x, y, x+w-1, y+hh-1)
              - sum_region(ii, x, y+hh, x+w-1, y+h-1))
    elif ftype == 2:
        tw = w // 3
        return (sum_region(ii, x, y, x+tw-1, y+h-1)
              - 2*sum_region(ii, x+tw, y, x+2*tw-1, y+h-1)
              + sum_region(ii, x+2*tw, y, x+w-1, y+h-1))
    elif ftype == 3:
        th = h // 3
        return (sum_region(ii, x, y, x+w-1, y+th-1)
              - 2*sum_region(ii, x, y+th, x+w-1, y+2*th-1)
              + sum_region(ii, x, y+2*th, x+w-1, y+h-1))
    else:
        hw, hh = w // 2, h // 2
        return (sum_region(ii, x, y, x+hw-1, y+hh-1)
              - sum_region(ii, x+hw, y, x+w-1, y+hh-1)
              - sum_region(ii, x, y+hh, x+hw-1, y+h-1)
              + sum_region(ii, x+hw, y+hh, x+w-1, y+h-1))


# ──────────────────────────────────────────────────────────────
# CARGA DE LA CASCADA
# ──────────────────────────────────────────────────────────────

def load_cascade(stage_dir: str, num_stages: int):
    """
    Carga hasta num_stages etapas desde archivos stage_NN_complete.pkl.
    Devuelve la lista de stages y el número real cargado.
    """
    stage_dir = Path(stage_dir)
    cascade   = []

    for i in range(num_stages):
        path = stage_dir / f'stage_{i+1:02d}_complete.pkl'
        if not path.exists():
            print(f'  ⚠️  stage_{i+1:02d}_complete.pkl no encontrado. Parando en {i} etapas.')
            break
        with open(path, 'rb') as f:
            stage = pickle.load(f)
        cascade.append(stage)
        n_clfs = len(stage['classifiers'])
        print(f'  ✅ Etapa {i+1:2d} cargada ({n_clfs} clasificadores débiles, θ={stage["theta"]:.4f})')

    if len(cascade) == 0:
        raise RuntimeError(f'No se encontró ninguna etapa en "{stage_dir}".')

    print(f'\n  → Cascada lista: {len(cascade)} etapas.\n')
    return cascade


# ──────────────────────────────────────────────────────────────
# PREDICCIÓN PARA UN PATCH (igual que en entrenamiento)
# ──────────────────────────────────────────────────────────────

def cascade_predict_patch(img_gray_24, cascade):
    """
    img_gray_24: imagen 24×24 uint8 en escala de grises.
    Devuelve 1 (cara) o 0 (no cara).
    """
    img = img_gray_24.astype(np.float64)
    h, w = img.shape
    ii  = integral_image(img)
    ii2 = integral_image(img ** 2)
    std = normalize_variance(ii, ii2, h, w)
    if std <= 0:
        std = 1.0

    for stage in cascade:
        score = 0.0
        for clf, alpha in zip(stage['classifiers'], stage['alphas']):
            fx, fy, fw, fh, ft = stage['features_global'][clf.feature_index]
            val = compute_feature(ii, fx, fy, fw, fh, ft) / std
            if clf.polarity * val < clf.polarity * clf.threshold:
                score += alpha
        if score < stage['theta']:
            return 0
    return 1


# ──────────────────────────────────────────────────────────────
# SLIDING WINDOW + PIRÁMIDE DE ESCALAS
# ──────────────────────────────────────────────────────────────

def detect_faces(frame_gray, cascade, min_size=24, max_size=None,
                 scale_factor=1.25, step_ratio=0.1):
    """
    Detecta caras con sliding window multi-escala.

    Parámetros
    ----------
    frame_gray   : imagen en escala de grises (cualquier tamaño).
    cascade      : lista de etapas de la cascada.
    min_size     : tamaño mínimo de ventana (píxeles).
    max_size     : tamaño máximo (None = min(H,W)//2).
    scale_factor : factor entre escalas (ej. 1.25).
    step_ratio   : paso de ventana como fracción del tamaño de ventana.

    Devuelve
    --------
    Lista de (x, y, w, h) con detecciones crudas (antes de NMS).
    """
    H, W = frame_gray.shape
    if max_size is None:
        max_size = min(H, W) // 2

    detections = []
    win_size = min_size

    while win_size <= max_size:
        step = max(1, int(win_size * step_ratio))
        scale = win_size / 24.0  # 24 es el tamaño de entrenamiento

        # Redimensionar el frame para que la ventana de 24px corresponda a win_size
        new_W = int(W / scale)
        new_H = int(H / scale)
        if new_W < 24 or new_H < 24:
            break

        resized = cv2.resize(frame_gray, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

        for y in range(0, new_H - 24 + 1, step):
            for x in range(0, new_W - 24 + 1, step):
                patch = resized[y:y+24, x:x+24]
                patch_norm = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                if cascade_predict_patch(patch_norm, cascade) == 1:
                    # Escalar coordenadas de vuelta al frame original
                    rx = int(x * scale)
                    ry = int(y * scale)
                    rw = int(win_size)
                    rh = int(win_size)
                    detections.append((rx, ry, rw, rh))

        win_size = int(win_size * scale_factor)

    return detections


# ──────────────────────────────────────────────────────────────
# NON-MAXIMUM SUPPRESSION
# ──────────────────────────────────────────────────────────────

def nms(detections, overlap_thresh=0.3):
    """
    NMS clásica por IoU. Devuelve lista de (x, y, w, h) finales.
    """
    if len(detections) == 0:
        return []

    boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in detections], dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]  # mayor área primero

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0, ix2 - ix1)
        inter_h = np.maximum(0, iy2 - iy1)
        inter   = inter_w * inter_h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= overlap_thresh]

    return [detections[i] for i in keep]


# ──────────────────────────────────────────────────────────────
# OVERLAY HUD
# ──────────────────────────────────────────────────────────────

def draw_hud(frame, fps, n_faces, stage_count):
    """Dibuja FPS, caras detectadas y número de etapas sobre el frame."""
    H, W = frame.shape[:2]

    # Fondo semitransparente para el HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 95), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Texto HUD
    font       = cv2.FONT_HERSHEY_SIMPLEX
    color_fps  = (0, 255, 120)   # verde neón
    color_face = (0, 200, 255)   # cyan
    color_stg  = (180, 180, 180) # gris

    cv2.putText(frame, f'FPS:    {fps:5.1f}', (10, 28),  font, 0.70, color_fps,  2, cv2.LINE_AA)
    cv2.putText(frame, f'Caras:  {n_faces}',  (10, 58),  font, 0.70, color_face, 2, cv2.LINE_AA)
    cv2.putText(frame, f'Etapas: {stage_count}',         (10, 85),  font, 0.55, color_stg,  1, cv2.LINE_AA)

    # Bounding boxes
    for (x, y, w, h) in _last_detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 255), 2)
        cv2.rectangle(frame, (x, y-18), (x+50, y), (0, 200, 255), -1)
        cv2.putText(frame, 'cara', (x+3, y-4), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


_last_detections = []  # estado global para el HUD entre frames


# ──────────────────────────────────────────────────────────────
# MODO IMAGEN ESTÁTICA
# ──────────────────────────────────────────────────────────────

def run_image(image_path, cascade, args):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'No se pudo leer la imagen: {image_path}')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    t0 = time.perf_counter()
    raw = detect_faces(gray, cascade,
                       min_size=args.min_size, max_size=args.max_size,
                       scale_factor=args.scale, step_ratio=args.step)
    final = nms(raw, overlap_thresh=args.nms_thresh)
    elapsed = time.perf_counter() - t0

    print(f'  Detecciones crudas: {len(raw)}  →  tras NMS: {len(final)}  ({elapsed*1000:.1f} ms)')

    for (x, y, w, h) in final:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'Caras: {len(final)}  ({elapsed*1000:.0f}ms)',
                (10, 30), font, 0.8, (0, 255, 120), 2, cv2.LINE_AA)

    out_path = Path(image_path).stem + '_detected.jpg'
    cv2.imwrite(out_path, img)
    print(f'  Resultado guardado en: {out_path}')
    cv2.imshow('Viola-Jones Detección', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
# MODO CÁMARA EN TIEMPO REAL
# ──────────────────────────────────────────────────────────────

def run_camera(cascade, args):
    global _last_detections

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir la cámara {args.camera}.')

    # Resolución de captura (puede reducirse para ganar velocidad)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'  Cámara abierta: {real_w}×{real_h}  |  Pulsa Q o ESC para salir.\n')

    fps_window = deque(maxlen=30)   # media móvil sobre los últimos 30 frames
    frame_count = 0
    detect_every = args.detect_every  # solo detectar 1 de cada N frames (=1 → todos)

    _last_detections = []
    n_faces = 0

    while True:
        t_frame = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print('Error leyendo frame. Fin.')
            break

        frame_count += 1

        # ── Detección (solo en los frames marcados) ─────────────
        if frame_count % detect_every == 0:
            # Opcionalmente reducir resolución para detección rápida
            if args.detect_scale < 1.0:
                dW = int(real_w * args.detect_scale)
                dH = int(real_h * args.detect_scale)
                detect_frame = cv2.resize(frame, (dW, dH))
                inv_s = 1.0 / args.detect_scale
            else:
                detect_frame = frame
                inv_s = 1.0

            gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

            raw   = detect_faces(gray, cascade,
                                 min_size=args.min_size, max_size=args.max_size,
                                 scale_factor=args.scale, step_ratio=args.step)
            final = nms(raw, overlap_thresh=args.nms_thresh)

            # Reescalar coordenadas al tamaño real del frame si se redujo
            if args.detect_scale < 1.0:
                final = [(int(x*inv_s), int(y*inv_s), int(w*inv_s), int(h*inv_s))
                         for (x, y, w, h) in final]

            _last_detections = final
            n_faces = len(final)

        # ── HUD ─────────────────────────────────────────────────
        elapsed = time.perf_counter() - t_frame
        fps_window.append(1.0 / elapsed if elapsed > 0 else 0)
        fps = float(np.mean(fps_window))

        frame = draw_hud(frame, fps, n_faces, len(cascade))

        cv2.imshow('Viola-Jones | Q/ESC = salir', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q o ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Cámara cerrada.')


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Inferencia Viola-Jones con cámara o imagen estática.'
    )

    # Cascada
    p.add_argument('--stages',    type=str, default='checkpoints_total',
                   help='Carpeta con los stage_NN_complete.pkl')
    p.add_argument('--num',       type=int, default=20,
                   help='Número máximo de etapas a cargar (se carga lo disponible)')

    # Detección
    p.add_argument('--scale',     type=float, default=1.25,
                   help='Factor de escala entre niveles de la pirámide (ej. 1.2)')
    p.add_argument('--step',      type=float, default=0.1,
                   help='Paso del sliding window como fracción del tamaño (ej. 0.1)')
    p.add_argument('--min-size',  type=int,   default=48,
                   help='Tamaño mínimo de ventana en píxeles')
    p.add_argument('--max-size',  type=int,   default=None,
                   help='Tamaño máximo de ventana (None = auto)')
    p.add_argument('--nms-thresh',type=float, default=0.3,
                   help='Umbral IoU para NMS (0.0–1.0)')

    # Modo cámara
    p.add_argument('--camera',       type=int,   default=0,
                   help='Índice de cámara (0 = por defecto)')
    p.add_argument('--cam-width',    type=int,   default=640,
                   help='Ancho de captura de cámara')
    p.add_argument('--cam-height',   type=int,   default=480,
                   help='Alto de captura de cámara')
    p.add_argument('--detect-every', type=int,   default=1,
                   help='Detectar 1 de cada N frames (1 = todos)')
    p.add_argument('--detect-scale', type=float, default=1.0,
                   help='Reducir frame antes de detectar (ej. 0.5 = mitad de resolución, más rápido)')

    # Modo imagen estática
    p.add_argument('--image', type=str, default=None,
                   help='Ruta a imagen estática (si se omite, usa la cámara)')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f'\n{"="*50}')
    print(f'  Viola-Jones Inference')
    print(f'  Cargando {args.num} etapa(s) desde "{args.stages}"')
    print(f'{"="*50}\n')

    cascade = load_cascade(args.stages, args.num)

    if args.image:
        run_image(args.image, cascade, args)
    else:
        run_camera(cascade, args)