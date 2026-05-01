"""
Viola-Jones Inference - Versión VECTORIZADA + NUMBA
=====================================================
Mejoras respecto a la versión anterior:
  - cascade_predict_batch portado a Numba (@njit parallel=True).
  - Cascada aplanada a arrays NumPy una sola vez al cargar (flatten_cascade).
  - Warm-up JIT automático al arrancar (cache=True para compilaciones futuras).
  - FPS calculado como tiempo entre frames consecutivos (corrección bug).
  - Sin cv2.NORM_MINMAX (normalización correcta por std local por patch).

Uso:
    python violajones_inference.py --stages checkpoints_total --num 12
    python violajones_inference.py --stages checkpoints_total --num 12 --scale 1.2 --step 0.1
    python violajones_inference.py --stages checkpoints_total --num 12 --images ( foto.jpg | data/images/ )
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
from numba import njit, prange
import numba as nb


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
# NÚCLEO NUMBA
# ──────────────────────────────────────────────────────────────

@njit(cache=True)
def _sum_region(ii, y1, x1, y2, x2):
    return ii[y2+1, x2+1] - ii[y1, x2+1] - ii[y2+1, x1] + ii[y1, x1]


@njit(cache=True)
def _compute_feature(ii, ox, oy, fw, fh, ftype, max_x, max_y):
    def cx(a): return min(a, max_x)
    def cy(a): return min(a, max_y)

    if ftype == 0:
        hw = fw // 2
        return (_sum_region(ii, oy, ox,        cy(oy+fh-1), cx(ox+hw-1))
              - _sum_region(ii, oy, cx(ox+hw), cy(oy+fh-1), cx(ox+fw-1)))
    elif ftype == 1:
        hh = fh // 2
        return (_sum_region(ii, oy,        ox, cy(oy+hh-1), cx(ox+fw-1))
              - _sum_region(ii, cy(oy+hh), ox, cy(oy+fh-1), cx(ox+fw-1)))
    elif ftype == 2:
        tw = fw // 3
        return (- _sum_region(ii, oy, ox,           cy(oy+fh-1), cx(ox+tw-1))
              + 2*_sum_region(ii, oy, cx(ox+tw),  cy(oy+fh-1), cx(ox+2*tw-1))
              - _sum_region(ii, oy, cx(ox+2*tw), cy(oy+fh-1), cx(ox+fw-1)))
    elif ftype == 3:
        th = fh // 3
        return (-_sum_region(ii, oy,         ox, cy(oy+th-1),   cx(ox+fw-1))
              + 2*_sum_region(ii, cy(oy+th),  ox, cy(oy+2*th-1), cx(ox+fw-1))
              -   _sum_region(ii, cy(oy+2*th), ox, cy(oy+fh-1),   cx(ox+fw-1)))
    else:
        hw = fw // 2
        hh = fh // 2
        return (_sum_region(ii, oy,        ox,        cy(oy+hh-1), cx(ox+hw-1))
              - _sum_region(ii, oy,        cx(ox+hw), cy(oy+hh-1), cx(ox+fw-1))
              - _sum_region(ii, cy(oy+hh), ox,        cy(oy+fh-1), cx(ox+hw-1))
              + _sum_region(ii, cy(oy+hh), cx(ox+hw), cy(oy+fh-1), cx(ox+fw-1)))


@njit(cache=True)
def cascade_predict_numba(ii, ii2,
                           xs, ys,
                           all_fx, all_fy, all_fw, all_fh, all_ft,
                           all_thresholds, all_polarities,
                           all_alphas,
                           stage_starts, stage_lengths,
                           stage_thetas):
    N   = xs.shape[0]
    WIN = 24
    max_y = ii.shape[0] - 2
    max_x = ii.shape[1] - 2
    n     = WIN * WIN

    alive = np.ones(N, dtype=nb.boolean)

    for p in prange(N):
        x = xs[p]
        y = ys[p]

        mu  = (ii[y+WIN, x+WIN] - ii[y, x+WIN] - ii[y+WIN, x] + ii[y, x]) / n
        ssq = (ii2[y+WIN, x+WIN] - ii2[y, x+WIN] - ii2[y+WIN, x] + ii2[y, x])
        var = ssq / n - mu * mu
        std = np.sqrt(max(var, 0.0))
        if std <= 0.0:
            std = 1.0

        for s in range(stage_starts.shape[0]):
            start  = stage_starts[s]
            length = stage_lengths[s]
            theta  = stage_thetas[s]

            score = 0.0
            for c in range(start, start + length):
                val = _compute_feature(ii,
                                       x + all_fx[c], y + all_fy[c],
                                       all_fw[c], all_fh[c], all_ft[c],
                                       max_x, max_y) / std
                if all_polarities[c] * val < all_polarities[c] * all_thresholds[c]:
                    score += all_alphas[c]

            if score < theta:
                alive[p] = False
                break

    return alive


# ──────────────────────────────────────────────────────────────
# APLANAR CASCADA PARA NUMBA
# ──────────────────────────────────────────────────────────────

def flatten_cascade(cascade):
    """Convierte la lista de stages en arrays planos para Numba."""
    fx_l, fy_l, fw_l, fh_l, ft_l = [], [], [], [], []
    thr_l, pol_l, alpha_l = [], [], []
    starts, lengths, thetas = [], [], []

    idx = 0
    for stage in cascade:
        clfs   = stage['classifiers']
        alphas = stage['alphas']
        feats  = stage['features_global']
        n = len(clfs)
        starts.append(idx)
        lengths.append(n)
        thetas.append(float(stage['theta']))
        for clf, alpha in zip(clfs, alphas):
            fx, fy, fw, fh, ft = feats[clf.feature_index]
            fx_l.append(int(fx)); fy_l.append(int(fy))
            fw_l.append(int(fw)); fh_l.append(int(fh)); ft_l.append(int(ft))
            thr_l.append(float(clf.threshold))
            pol_l.append(float(clf.polarity))
            alpha_l.append(float(alpha))
        idx += n

    return (np.array(fx_l,    np.int32),
            np.array(fy_l,    np.int32),
            np.array(fw_l,    np.int32),
            np.array(fh_l,    np.int32),
            np.array(ft_l,    np.int32),
            np.array(thr_l,   np.float64),
            np.array(pol_l,   np.float64),
            np.array(alpha_l, np.float64),
            np.array(starts,  np.int32),
            np.array(lengths, np.int32),
            np.array(thetas,  np.float64))


# ──────────────────────────────────────────────────────────────
# WRAPPER CASCADE PREDICT
# ──────────────────────────────────────────────────────────────

def cascade_predict_batch(ii, ii2, xs, ys, flat):
    return cascade_predict_numba(ii, ii2, xs, ys, *flat)


# ──────────────────────────────────────────────────────────────
# DETECCIÓN MULTI-ESCALA VECTORIZADA
# ──────────────────────────────────────────────────────────────

def detect_faces_fast(frame_gray, flat,
                      min_size=48, max_size=None,
                      scale_factor=1.25, step_ratio=0.1,
                      n_threads=1):
    H, W = frame_gray.shape
    if max_size is None:
        max_size = min(H, W) // 2

    all_detections = []

    def process_scale(win_size):
        scale = win_size / 24.0
        new_W = int(W / scale)
        new_H = int(H / scale)
        if new_W < 24 or new_H < 24:
            return []

        resized = cv2.resize(frame_gray, (new_W, new_H),
                             interpolation=cv2.INTER_LINEAR).astype(np.float64)

        ii  = integral_image(resized)
        ii2 = integral_image(resized ** 2)

        step = max(1, round(24 * step_ratio))

        ys_arr = np.arange(0, new_H - 24 + 1, step, dtype=np.int32)
        xs_arr = np.arange(0, new_W - 24 + 1, step, dtype=np.int32)
        yy, xx = np.meshgrid(ys_arr, xs_arr, indexing='ij')
        xs = xx.ravel()
        ys = yy.ravel()

        if len(xs) == 0:
            return []

        alive = cascade_predict_batch(ii, ii2, xs, ys, flat)

        detected = []
        for x, y in zip(xs[alive], ys[alive]):
            rx = int(x * scale)
            ry = int(y * scale)
            detected.append((rx, ry, int(win_size), int(win_size)))
        return detected

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
    print(f'\n  → Cascada lista: {len(cascade)} etapas.')

    print('  → Aplanando cascada para Numba...')
    flat = flatten_cascade(cascade)
    print('  → Compilando JIT (solo la primera vez, puede tardar ~10s)...')
    dummy_ii  = np.zeros((26, 26), np.float64)
    dummy_ii2 = np.zeros((26, 26), np.float64)
    dummy_xs  = np.array([0], np.int32)
    dummy_ys  = np.array([0], np.int32)
    cascade_predict_numba(dummy_ii, dummy_ii2, dummy_xs, dummy_ys, *flat)
    print('  → JIT listo.\n')

    return cascade, flat


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

def run_image(image_path, flat, cascade, args):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'No se pudo leer: {image_path}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0   = time.perf_counter()
    raw  = detect_faces_fast(gray, flat,
                             min_size=args.min_size, max_size=args.max_size,
                             scale_factor=args.scale, step_ratio=args.step,
                             n_threads=args.threads)
    final   = nms(raw, overlap_thresh=args.nms_thresh)
    elapsed = time.perf_counter() - t0
    print(f'  Crudas: {len(raw)}  →  NMS: {len(final)}  ({elapsed*1000:.1f} ms)')
    for (x, y, w, h) in final:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 255), 2)
    cv2.putText(img, f'Caras: {len(final)}  ({elapsed*1000:.0f}ms)',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,120), 2, cv2.LINE_AA)
    out = Path(image_path).stem + '_detected.jpg'
    cv2.imwrite(out, img)
    print(f'  Guardado en: {out}')
    cv2.imshow('Viola-Jones Detección', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
# MODO CÁMARA
# ──────────────────────────────────────────────────────────────

def run_camera(cascade, flat, args):
    global _last_detections

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir la cámara {args.camera}.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'  Cámara: {real_w}×{real_h}  |  Q/ESC = salir\n')

    fps_window       = deque(maxlen=30)
    frame_count      = 0
    n_faces          = 0
    _last_detections = []
    last_time        = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # FPS real: tiempo entre frames consecutivos
        now       = time.perf_counter()
        elapsed   = now - last_time
        last_time = now
        fps_window.append(1.0 / elapsed if elapsed > 0 else 0)
        fps = float(np.mean(fps_window))

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
            raw   = detect_faces_fast(gray, flat,
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
    p = argparse.ArgumentParser(description='Viola-Jones Inference (Numba)')
    p.add_argument('--stages',       type=str,   default='checkpoints_total')
    p.add_argument('--num',          type=int,   default=20)
    p.add_argument('--scale',        type=float, default=1.25)
    p.add_argument('--step',         type=float, default=0.1)
    p.add_argument('--min-size',     type=int,   default=24)
    p.add_argument('--max-size',     type=int,   default=None)
    p.add_argument('--nms-thresh',   type=float, default=0.3)
    p.add_argument('--camera',       type=int,   default=0)
    p.add_argument('--cam-width',    type=int,   default=640)
    p.add_argument('--cam-height',   type=int,   default=480)
    p.add_argument('--detect-every', type=int,   default=1)
    p.add_argument('--detect-scale', type=float, default=1.0)
    p.add_argument('--threads',      type=int,   default=1)
    p.add_argument('--images',        type=str,   default=None)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    print(f'\n{"="*50}')
    print(f'  Viola-Jones Inference (NUMBA)')
    print(f'  Cargando {args.num} etapa(s) desde "{args.stages}"')
    print(f'{"="*50}\n')
    cascade, flat = load_cascade(args.stages, args.num)
    if args.images:
        if os.path.isdir(args.images):
            img_files = [f for f in os.listdir(args.images)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if not img_files:
                print(f'  No se encontraron imágenes en "{args.images}".')
            else:
                print(f'  Procesando {len(img_files)} imagen(es) en "{args.images}"...\n')
                for img_file in img_files:
                    print(f'  → {img_file}')
                    run_image(os.path.join(args.images, img_file), flat, cascade, args)
        else:
            run_image(args.images, flat, cascade, args)
    else:
        run_camera(cascade, flat, args)