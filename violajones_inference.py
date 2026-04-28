"""
Viola-Jones Inference - Versión VECTORIZADA (rápida)
=====================================================
Diferencias clave respecto a la versión original:
  - detect_faces vectorizado: calcula integral image una vez por escala y
    evalúa TODOS los patches simultáneamente con slicing NumPy.
  - cascade_predict_batch: evalúa toda una matriz de patches de golpe,
    eliminando el bucle Python patch-a-patch.
  - Sin cv2.normalize por patch (se usa la std de la integral image directamente).
  - Multithreading opcional por escala (--threads N).

Uso:
    python inference_vj_fast.py --stages checkpoints_total --num 12
    python inference_vj_fast.py --stages checkpoints_total --num 12 --scale 1.2 --step 0.1
    python inference_vj_fast.py --stages checkpoints_total --num 12 --image foto.jpg
"""

import os
import cv2
import pickle
import argparse
import time
import numpy as np
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor


# ──────────────────────────────────────────────────────────────
# WEAK CLASSIFIER
# ──────────────────────────────────────────────────────────────

class WeakClassifier:
    __slots__ = ('feature_index', 'threshold', 'polarity')
    def __init__(self, feature_index, threshold, polarity):
        self.feature_index = feature_index
        self.threshold     = threshold
        self.polarity      = polarity


# ──────────────────────────────────────────────────────────────
# INTEGRAL IMAGE
# ──────────────────────────────────────────────────────────────

def integral_image(img):
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)
    return ii


# ──────────────────────────────────────────────────────────────
# SUMA DE REGIÓN VECTORIZADA SOBRE TODOS LOS PATCHES
# ──────────────────────────────────────────────────────────────

def sum_region_batch(ii, x1, y1, x2, y2):
    """
    Suma de región usando la integral image para N patches a la vez.
    x1, y1: coordenadas absolutas top-left (arrays int32).
    x2, y2: coordenadas absolutas bottom-right INCLUSIVE (arrays int32).
    Los índices ya vienen cliqueados a los límites de ii en cascade_predict_batch.
    """
    return (ii[y2 + 1, x2 + 1]
          - ii[y1,     x2 + 1]
          - ii[y2 + 1, x1    ]
          + ii[y1,     x1    ])


# ──────────────────────────────────────────────────────────────
# FEATURE VECTORIZADO
# ──────────────────────────────────────────────────────────────

def compute_feature_batch(ii, ox, oy, fw, fh, ftype, max_x, max_y):
    """
    Calcula un Haar feature para N patches a la vez.
    ox, oy  : coordenada absoluta del origen de la feature en la imagen (arrays).
    fw, fh  : tamaño de la feature en píxeles.
    max_x, max_y: límites máximos de coordenada (new_W-1, new_H-1).
    Devuelve array float64 de shape (N,).

    Nota: ox = patch_x + feature_x_dentro_del_patch (ya sumado antes de llamar).
    """
    # Clipear para que ningún índice se salga de la imagen
    def clip_x(a): return np.minimum(a, max_x)
    def clip_y(a): return np.minimum(a, max_y)

    if ftype == 0:          # two-rect horizontal
        hw = fw // 2
        return (sum_region_batch(ii, ox,      oy,          clip_x(ox+hw-1),    clip_y(oy+fh-1))
              - sum_region_batch(ii, clip_x(ox+hw), oy,    clip_x(ox+fw-1),    clip_y(oy+fh-1)))
    elif ftype == 1:        # two-rect vertical
        hh = fh // 2
        return (sum_region_batch(ii, ox, oy,               clip_x(ox+fw-1), clip_y(oy+hh-1))
              - sum_region_batch(ii, ox, clip_y(oy+hh),    clip_x(ox+fw-1), clip_y(oy+fh-1)))
    elif ftype == 2:        # three-rect horizontal
        tw = fw // 3
        return (sum_region_batch(ii, ox,               oy, clip_x(ox+tw-1),    clip_y(oy+fh-1))
              - 2*sum_region_batch(ii, clip_x(ox+tw),  oy, clip_x(ox+2*tw-1),  clip_y(oy+fh-1))
              + sum_region_batch(ii, clip_x(ox+2*tw),  oy, clip_x(ox+fw-1),    clip_y(oy+fh-1)))
    elif ftype == 3:        # three-rect vertical
        th = fh // 3
        return (sum_region_batch(ii, ox, oy,               clip_x(ox+fw-1), clip_y(oy+th-1))
              - 2*sum_region_batch(ii, ox, clip_y(oy+th),  clip_x(ox+fw-1), clip_y(oy+2*th-1))
              + sum_region_batch(ii, ox, clip_y(oy+2*th),  clip_x(ox+fw-1), clip_y(oy+fh-1)))
    else:                   # four-rect (checker)
        hw, hh = fw // 2, fh // 2
        return (sum_region_batch(ii, ox,               oy,             clip_x(ox+hw-1), clip_y(oy+hh-1))
              - sum_region_batch(ii, clip_x(ox+hw),    oy,             clip_x(ox+fw-1), clip_y(oy+hh-1))
              - sum_region_batch(ii, ox,               clip_y(oy+hh),  clip_x(ox+hw-1), clip_y(oy+fh-1))
              + sum_region_batch(ii, clip_x(ox+hw),    clip_y(oy+hh),  clip_x(ox+fw-1), clip_y(oy+fh-1)))


# ──────────────────────────────────────────────────────────────
# PREDICCIÓN BATCH — NÚCLEO DE LA OPTIMIZACIÓN
# ──────────────────────────────────────────────────────────────

def cascade_predict_batch(ii, ii2, xs, ys, cascade):
    """
    Evalúa la cascada para todos los patches (xs[i], ys[i]) a la vez.

    ii, ii2 : integral images (float64) de la imagen redimensionada.
              shape = (H+1, W+1), así que el índice válido es 0..H / 0..W
    xs, ys  : arrays int32 con coordenadas top-left de cada patch 24×24.
    cascade : lista de stages cargados.

    Devuelve array bool de shape (N,) — True = cara detectada.
    """
    N   = len(xs)
    WIN = 24

    # Límites seguros para indexar ii (shape = H+1 x W+1)
    max_y = ii.shape[0] - 2   # índice máximo para y2 (luego se hace y2+1)
    max_x = ii.shape[1] - 2   # índice máximo para x2

    # Std por patch usando la integral cuadrática
    n   = WIN * WIN
    mu  = (ii[ys+WIN, xs+WIN] - ii[ys, xs+WIN] - ii[ys+WIN, xs] + ii[ys, xs]) / n
    ssq = (ii2[ys+WIN, xs+WIN] - ii2[ys, xs+WIN] - ii2[ys+WIN, xs] + ii2[ys, xs])
    var = ssq / n - mu**2
    std = np.sqrt(np.maximum(var, 0))
    std[std <= 0] = 1.0

    alive = np.ones(N, dtype=bool)   # patches que todavía no fueron rechazados

    for stage in cascade:
        if not alive.any():
            break

        # Índices de patches activos
        idx = np.where(alive)[0]
        xa, ya, stda = xs[idx], ys[idx], std[idx]

        score = np.zeros(len(idx), dtype=np.float64)

        for clf, alpha in zip(stage['classifiers'], stage['alphas']):
            fx, fy, fw, fh, ft = stage['features_global'][clf.feature_index]
            # ox/oy = coordenada absoluta del origen de la feature en la imagen
            ox = xa + fx
            oy = ya + fy
            vals = compute_feature_batch(ii, ox, oy, fw, fh, ft, max_x, max_y) / stda
            pos = (clf.polarity * vals < clf.polarity * float(clf.threshold))
            score += np.where(pos, alpha, 0.0)

        # Rechazar patches que no superan el umbral
        alive[idx[score < stage['theta']]] = False

    return alive


# ──────────────────────────────────────────────────────────────
# DETECCIÓN MULTI-ESCALA VECTORIZADA
# ──────────────────────────────────────────────────────────────

def detect_faces_fast(frame_gray, cascade,
                      min_size=48, max_size=None,
                      scale_factor=1.25, step_ratio=0.1,
                      n_threads=1):
    """
    Detección con sliding window + pirámide de escalas, completamente vectorizada.
    """
    H, W = frame_gray.shape
    if max_size is None:
        max_size = min(H, W) // 2

    all_detections = []
    win_size = min_size

    def process_scale(win_size):
        scale = win_size / 24.0
        new_W = int(W / scale)
        new_H = int(H / scale)
        if new_W < 24 or new_H < 24:
            return []

        resized = cv2.resize(frame_gray, (new_W, new_H),
                             interpolation=cv2.INTER_LINEAR).astype(np.float64)
        
        # resized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)

        ii  = integral_image(resized)
        ii2 = integral_image(resized ** 2)

        step = max(1, round(24 * step_ratio))

        # Generar todas las posiciones de patch de golpe
        ys_arr = np.arange(0, new_H - 24 + 1, step, dtype=np.int32)
        xs_arr = np.arange(0, new_W - 24 + 1, step, dtype=np.int32)
        yy, xx = np.meshgrid(ys_arr, xs_arr, indexing='ij')
        xs = xx.ravel()
        ys = yy.ravel()

        if len(xs) == 0:
            return []

        alive = cascade_predict_batch(ii, ii2, xs, ys, cascade)

        # Escalar coords de vuelta al frame original
        detected = []
        for x, y in zip(xs[alive], ys[alive]):
            rx = int(x * scale)
            ry = int(y * scale)
            rw = int(win_size)
            rh = int(win_size)
            detected.append((rx, ry, rw, rh))
        return detected

    # Construir lista de escalas
    scales = []
    ws = min_size
    while ws <= max_size:
        scales.append(ws)
        ws = int(ws * scale_factor)

    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            results = list(ex.map(process_scale, scales))
        for r in results:
            all_detections.extend(r)
    else:
        for ws in scales:
            all_detections.extend(process_scale(ws))

    return all_detections


# ──────────────────────────────────────────────────────────────
# NMS
# ──────────────────────────────────────────────────────────────

def nms(detections, overlap_thresh=0.3):
    if len(detections) == 0:
        return []
    boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in detections], dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= overlap_thresh]
    return [detections[i] for i in keep]


# ──────────────────────────────────────────────────────────────
# CARGA DE CASCADA
# ──────────────────────────────────────────────────────────────

def load_cascade(stage_dir: str, num_stages: int):
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
    if not cascade:
        raise RuntimeError(f'No se encontró ninguna etapa en "{stage_dir}".')
    print(f'\n  → Cascada lista: {len(cascade)} etapas.\n')
    return cascade


# ──────────────────────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────────────────────

_last_detections = []

def draw_hud(frame, fps, n_faces, stage_count):
    H, W = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 95), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'FPS:    {fps:5.1f}', (10, 28), font, 0.70, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Caras:  {n_faces}',  (10, 58), font, 0.70, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Etapas: {stage_count}', (10, 85), font, 0.55, (180,180,180), 1, cv2.LINE_AA)
    for (x, y, w, h) in _last_detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 255), 2)
        cv2.rectangle(frame, (x, y-18), (x+50, y), (0, 200, 255), -1)
        cv2.putText(frame, 'cara', (x+3, y-4), font, 0.45, (0,0,0), 1, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────────────────────
# MODO IMAGEN ESTÁTICA
# ──────────────────────────────────────────────────────────────

def run_camera(cascade, args):
    global _last_detections

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir la cámara {args.camera}.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'  Cámara: {real_w}×{real_h}  |  Q/ESC = salir\n')

    fps_window  = deque(maxlen=30)
    frame_count = 0
    n_faces     = 0
    _last_detections = []
    last_time = time.perf_counter()   # ← AÑADIDO

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ── FPS real: tiempo entre frames consecutivos ──
        now     = time.perf_counter()
        elapsed = now - last_time
        last_time = now
        fps_window.append(1.0 / elapsed if elapsed > 0 else 0)
        fps = float(np.mean(fps_window))
        # ────────────────────────────────────────────────

        if frame_count % args.detect_every == 0:
            if args.detect_scale < 1.0:
                dW = int(real_w * args.detect_scale)
                dH = int(real_h * args.detect_scale)
                detect_frame = cv2.resize(frame, (dW, dH))
                inv_s = 1.0 / args.detect_scale
            else:
                detect_frame = frame
                inv_s = 1.0

            gray  = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
            raw   = detect_faces_fast(gray, cascade,
                                      min_size=args.min_size, max_size=args.max_size,
                                      scale_factor=args.scale, step_ratio=args.step,
                                      n_threads=args.threads)
            final = nms(raw, overlap_thresh=args.nms_thresh)

            if args.detect_scale < 1.0:
                final = [(int(x*inv_s), int(y*inv_s), int(w*inv_s), int(h*inv_s))
                         for (x, y, w, h) in final]

            _last_detections = final
            n_faces = len(final)

        frame = draw_hud(frame, fps, n_faces, len(cascade))
        cv2.imshow('Viola-Jones Fast | Q/ESC = salir', frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Cámara cerrada.')


# ──────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Viola-Jones Inference (vectorizada)')
    p.add_argument('--stages',       type=str,   default='checkpoints_total')
    p.add_argument('--num',          type=int,   default=20)
    p.add_argument('--scale',        type=float, default=1.25)
    p.add_argument('--step',         type=float, default=0.1)
    p.add_argument('--min-size',     type=int,   default=48)
    p.add_argument('--max-size',     type=int,   default=None)
    p.add_argument('--nms-thresh',   type=float, default=0.3)
    p.add_argument('--camera',       type=int,   default=0)
    p.add_argument('--cam-width',    type=int,   default=640)
    p.add_argument('--cam-height',   type=int,   default=480)
    p.add_argument('--detect-every', type=int,   default=1)
    p.add_argument('--detect-scale', type=float, default=1.0)
    p.add_argument('--threads',      type=int,   default=1,
                   help='Threads para procesar escalas en paralelo (ej. 4)')
    p.add_argument('--image',        type=str,   default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f'\n{"="*50}')
    print(f'  Viola-Jones Inference (FAST)')
    print(f'  Cargando {args.num} etapa(s) desde "{args.stages}"')
    print(f'{"="*50}\n')
    cascade = load_cascade(args.stages, args.num)
    if args.image:
        run_image(args.image, cascade, args)
    else:
        run_camera(cascade, args)