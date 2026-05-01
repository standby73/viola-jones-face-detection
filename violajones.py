"""
Viola-Jones (Haar H/V) - Versión Dinámica
=========================================
Cascada de clasificadores AdaBoost con features Haar horizontales y verticales.
Checkpointing granular.
- Carga de positivos con Data Augmentation automático (Espejo).
- Mining dinámico de negativos a partir de imágenes de fondo grandes.
- Solución a errores de memoria/pickle usando backend de threading.

CAMBIOS:
  - Las features Haar NO se guardan en pkl (se regeneran en memoria, son deterministas).
  - El conjunto de TEST es estático: caras desde PATH_FACES_TEST y crops negativos
    desde PATH_BACKGROUNDS_TEST. No hay train_test_split.
"""

import os

import cv2
import numpy as np
import random
import pickle
import time
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix

# ──────────────────────────────────────────────────────────────
# CARGA DE IMÁGENES
# ──────────────────────────────────────────────────────────────

def load_background_paths(path_bg):
    """
    Busca imágenes grandes de fondo y devuelve solo sus rutas.
    Evita colapsar la RAM cargando todas las matrices de píxeles.
    """
    print('Escaneando rutas de imágenes de fondo...')
    bg_files = []

    for ext in ('*.jpg', '*.png', '*.pgm', '*.jpeg', '*.JPG', '*.PNG'):
        bg_files.extend(Path(path_bg).rglob(ext))

    bg_paths = list(set([str(p) for p in bg_files]))

    print(f'Dataset Fondos: {len(bg_paths)} rutas listas para lectura dinámica.')
    return bg_paths

def load_images(path, is_negative=False, test=False):
    """Carga imágenes positivas o negativas. Si es positivo y no es test, añade versión Flip."""
    label = 'Negativos' if is_negative else 'Positivos'
    mode = 'Test' if test else 'Train'
    print(f'Cargando imágenes {label} ({mode})...')

    files = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]

    images = []
    for fp in tqdm(files, desc=f'Cargando {label} {mode}'):
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img.astype(np.uint8))
            if not is_negative and not test:
                images.append(cv2.flip(img, 1))

    suffix = f' (incluyendo su respectivo FLip)' if not is_negative and not test else ''
    print(f'Dataset {label} {mode}: {len(images)} imágenes{suffix}.')
    return images

# ──────────────────────────────────────────────────────────────
# NORMALIZACIÓN, CALCULO INTEGRAL y AUMENTADOS DINÁMICOS
# ──────────────────────────────────────────────────────────────

def normalize_variance(ii, ii2, h, w):
    """
    Calcula σ usando las dos integrales ya calculadas.
    σ² = E[x²] - E[x]²
    """
    N      = h * w
    mean   = (ii[h, w] - ii[0, w] - ii[h, 0] + ii[0, 0]) / N
    sum_sq = ii2[h, w] - ii2[0, w] - ii2[h, 0] + ii2[0, 0]
    var    = (sum_sq / N) - (mean ** 2)
    return np.sqrt(max(var, 0))

def integral_image(img):
    ii = np.zeros((img.shape[0]+1, img.shape[1]+1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.int64), axis=0), axis=1)
    return ii

def apply_dynamic_augmentation(img):
    working = img.clip(0, 255).astype(np.uint8)
    op = random.randint(0, 2)
    if op == 0:
        return working
    h, w = working.shape
    if op == 1:
        M = cv2.getRotationMatrix2D((w//2, h//2), random.uniform(-10, 10), 1)
        return cv2.warpAffine(working, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    else:
        M = np.float32([[1, random.uniform(-0.1, 0.1), 0], [0, 1, 0]])
        return cv2.warpAffine(working, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ──────────────────────────────────────────────────────────────
# VALIDACIÓN Y MINING DE NEGATIVOS DINÁMICOS
# ──────────────────────────────────────────────────────────────

def is_valid_crop(crop_gray):
    """Rechaza crops demasiado uniformes (cielo blanco, paredes lisas)."""
    std = crop_gray.std()
    mean = crop_gray.mean()
    if std < 15: return False
    if mean < 20: return False
    if mean > 235: return False
    if np.mean(crop_gray > 240) > 0.6: return False
    return True

def _mine_dynamic_negatives(bg_paths, cascade, required_n, crop_size=24):
    """
    Lee imágenes dinámicamente desde el disco. Extrae lotes de crops
    por imagen para no saturar el disco duro con lecturas continuas.
    """
    fps = []
    attempts = 0
    max_attempts = required_n * 2000

    CROPS_PER_IMAGE = 100

    desc_str = 'Mining False Positives' if len(cascade) > 0 else 'Mining Random Crops'
    pbar = tqdm(total=required_n, desc=f'  {desc_str}', leave=False)

    while len(fps) < required_n and attempts < max_attempts:

        bg_path = random.choice(bg_paths)
        bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)

        if bg is None:
            continue

        h, w = bg.shape
        if h <= crop_size or w <= crop_size:
            continue

        for _ in range(CROPS_PER_IMAGE):

            if len(fps) >= required_n or attempts >= max_attempts:
                break

            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)

            crop = bg[y:y + crop_size, x:x + crop_size]

            crop_final = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA).astype(np.uint8)

            if not is_valid_crop(crop_final):
                attempts += 1
                continue

            if len(cascade) != 0:
                if cascade_predict_online(crop_final, cascade) == 1:
                    fps.append(crop_final)
                    pbar.update(1)
            else:
                fps.append(crop_final)
                pbar.update(1)

            attempts += 1

    pbar.close()

    if len(fps) < required_n:
        print(f'  ⚠️ Solo {len(fps)}/{required_n} tras {attempts} intentos.')
    else:
        print(f'  ✅ Minados {len(fps)} negativos en {attempts} intentos.')

    return fps


# ──────────────────────────────────────────────────────────────
# FEATURES DE HAAR
# ──────────────────────────────────────────────────────────────

def sum_region(ii, x1, y1, x2, y2):
    """Suma de región en integral image. Acepta escalares o arrays."""
    return ii[y2+1, x2+1] - ii[y1, x2+1] - ii[y2+1, x1] + ii[y1, x1]

def compute_feature(ii, x, y, w, h, ftype=0):
    if ftype == 0:
        hw = w // 2
        return (sum_region(ii, x, y, x+hw-1, y+h-1) - sum_region(ii, x+hw, y, x+w-1, y+h-1))
    elif ftype == 1:
        hh = h // 2
        return (sum_region(ii, x, y, x+w-1, y+hh-1) - sum_region(ii, x, y+hh, x+w-1, y+h-1))
    elif ftype == 2:
        tw = w // 3
        return (- sum_region(ii, x, y, x+tw-1, y+h-1) + 2*sum_region(ii, x+tw, y, x+2*tw-1, y+h-1) - sum_region(ii, x+2*tw, y, x+w-1, y+h-1))
    elif ftype == 3:
        th = h // 3
        return (- sum_region(ii, x, y, x+w-1, y+th-1) + 2*sum_region(ii, x, y+th, x+w-1, y+2*th-1) - sum_region(ii, x, y+2*th, x+w-1, y+h-1))
    else:
        hw = w // 2
        hh = h // 2
        return (sum_region(ii, x, y, x+hw-1, y+hh-1) - sum_region(ii, x+hw, y, x+w-1, y+hh-1) - sum_region(ii, x, y+hh, x+hw-1, y+h-1) + sum_region(ii, x+hw, y+hh, x+w-1, y+h-1))

def generate_features(img_shape, step=1, max_features=None, seed=42):
    """
    Genera features Haar de forma determinista (misma seed → mismas features).
    NO se guarda en disco: se regenera en memoria cada vez que se necesita.
    """
    rng = random.Random(seed)
    H, W = img_shape
    type_constraints = {
        0: (6, 4, 'w%2'), 1: (4, 6, 'h%2'), 2: (6, 4, 'w%3'),
        3: (4, 6, 'h%3'), 4: (4, 4, 'wh%2'),
    }
    all_features = []

    for ftype, (min_w, min_h, div) in type_constraints.items():
        bucket = []
        for width in range(min_w, W + 1, step):
            for height in range(min_h, H + 1, step):
                if div == 'w%2'  and width  % 2 != 0: continue
                if div == 'h%2'  and height % 2 != 0: continue
                if div == 'w%3'  and width  % 3 != 0: continue
                if div == 'h%3'  and height % 3 != 0: continue
                if div == 'wh%2' and (width % 2 != 0 or height % 2 != 0): continue
                for x in range(0, W - width + 1, step):
                    for y in range(0, H - height + 1, step):
                        bucket.append((x, y, width, height, ftype))

        if max_features is None:
            all_features.extend(bucket)
        else:
            max_per_type = max_features // len(type_constraints)
            sample = bucket if len(bucket) <= max_per_type else rng.sample(bucket, max_per_type)
            all_features.extend(sample)

    rng.shuffle(all_features)
    print(f'Generadas {len(all_features):,} características Haar (5 tipos).')
    return all_features


# ──────────────────────────────────────────────────────────────
# EXTRACCIÓN DE FEATURES A DISCO (memmap) (PROBLEMAS DE RAM)
# ──────────────────────────────────────────────────────────────

def _compute_all_features_vectorized(ii, features_np):
    """
    features_np: np.array shape (N, 5) con columnas [x, y, w, h, ftype]
    Devuelve: np.array shape (N,) con el valor de cada feature.
    """

    x  = features_np[:, 0].astype(np.int32)
    y  = features_np[:, 1].astype(np.int32)
    w  = features_np[:, 2].astype(np.int32)
    h  = features_np[:, 3].astype(np.int32)
    ft = features_np[:, 4].astype(np.int32)

    results = np.zeros(len(features_np), dtype=np.int64)

    mask = ft == 0
    if mask.any():
        hw = w[mask] // 2
        results[mask] = sum_region(ii, x[mask], y[mask], x[mask]+hw-1, y[mask]+h[mask]-1) \
                        - sum_region(ii, x[mask]+hw, y[mask], x[mask]+w[mask]-1, y[mask]+h[mask]-1)

    mask = ft == 1
    if mask.any():
        hh = h[mask] // 2
        results[mask] = sum_region(ii, x[mask], y[mask], x[mask]+w[mask]-1, y[mask]+hh-1) \
                        - sum_region(ii, x[mask], y[mask]+hh, x[mask]+w[mask]-1, y[mask]+h[mask]-1)

    mask = ft == 2
    if mask.any():
        tw = w[mask] // 3
        results[mask] = - sum_region(ii, x[mask],      y[mask], x[mask]+tw-1,    y[mask]+h[mask]-1) \
                        + 2*sum_region(ii, x[mask]+tw,   y[mask], x[mask]+2*tw-1,  y[mask]+h[mask]-1) \
                        - sum_region(ii, x[mask]+2*tw, y[mask], x[mask]+w[mask]-1, y[mask]+h[mask]-1)

    mask = ft == 3
    if mask.any():
        th = h[mask] // 3
        results[mask] = - sum_region(ii, x[mask], y[mask],      x[mask]+w[mask]-1, y[mask]+th-1) \
                        + 2*sum_region(ii, x[mask], y[mask]+th,   x[mask]+w[mask]-1, y[mask]+2*th-1) \
                        - sum_region(ii, x[mask], y[mask]+2*th, x[mask]+w[mask]-1, y[mask]+h[mask]-1)

    mask = ft == 4
    if mask.any():
        hw, hh = w[mask]//2, h[mask]//2
        results[mask] = sum_region(ii, x[mask],    y[mask],    x[mask]+hw-1,      y[mask]+hh-1) \
                        - sum_region(ii, x[mask]+hw, y[mask],    x[mask]+w[mask]-1, y[mask]+hh-1) \
                        - sum_region(ii, x[mask],    y[mask]+hh, x[mask]+hw-1,      y[mask]+h[mask]-1) \
                        + sum_region(ii, x[mask]+hw, y[mask]+hh, x[mask]+w[mask]-1, y[mask]+h[mask]-1)

    return results.astype(np.float64)

def _process_chunk_to_array(args):
    imgs, features_arr, augment, start_idx = args
    features_np = np.array(features_arr, dtype=np.int32)
    results = np.zeros((len(imgs), len(features_arr)), dtype=np.float32)

    for i, img in enumerate(imgs):
        if augment:
            img = apply_dynamic_augmentation(img)
        img = img.astype(np.float64)
        h,w = img.shape
        ii = integral_image(img)
        ii2 = integral_image(img**2)
        std = normalize_variance(ii, ii2, h, w)
        if std <= 0:
            std = 1.0
        results[i] = _compute_all_features_vectorized(ii, features_np) / std

    return start_idx, results

def _fill_mmap_chunk(X_list, features, mmap_path, mmap_shape, row_offset=0, augment=False, n_jobs=-1, desc=''):
    features_arr = list(features)
    n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    chunk_size = max(50, len(X_list) // (n_workers * 4))

    chunks = []
    for start in range(0, len(X_list), chunk_size):
        end = min(start + chunk_size, len(X_list))
        chunks.append((X_list[start:end], features_arr, augment, row_offset + start))

    print(f'  Extrayendo {desc}: {len(X_list)} imgs (chunk_size={chunk_size})', flush=True)

    mmap = np.memmap(mmap_path, dtype=np.float32, mode='r+', shape=mmap_shape)

    try:
        results = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=0)(
            delayed(_process_chunk_to_array)(chunk)
            for chunk in tqdm(chunks, desc=f'  {desc}')
        )
    except Exception as e:
        print("Salto a hilos")
        results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
            delayed(_process_chunk_to_array)(chunk)
            for chunk in tqdm(chunks, desc=f'  {desc}')
        )

    for start_idx, arr in results:
        mmap[start_idx:start_idx + arr.shape[0], :] = arr

    mmap.flush()
    del mmap


# ──────────────────────────────────────────────────────────────
# 5. CLASIFICADORES Y ADABOOST
# ──────────────────────────────────────────────────────────────

class WeakClassifier:
    __slots__ = ('feature_index', 'threshold', 'polarity')
    def __init__(self, feature_index, threshold, polarity):
        self.feature_index = feature_index
        self.threshold     = threshold
        self.polarity      = polarity

    def predict(self, X):
        feat = np.array(X[:, self.feature_index], dtype=np.float64)
        return (self.polarity * feat < self.polarity * float(self.threshold)).astype(np.int32)

def adaboost_round(X, y, weights, mmap_path, mmap_shape):
    mmap = np.memmap(mmap_path, dtype=np.float32, mode='r', shape=mmap_shape)
    X_ram = np.array(mmap)
    del mmap

    n_features = X_ram.shape[1]
    y_bool = np.asarray(y, dtype=bool)
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()

    pos_mask = np.asarray(y, dtype=np.int32) == 1
    neg_mask = ~pos_mask

    t0 = time.time()

    def eval_chunk(f_start, f_end):
        chunk = np.array(X_ram[:, f_start:f_end], dtype=np.int64)
        best_err, best_f, best_thr, best_pol = np.inf, 0, 0.0, 1

        for i, f_idx in enumerate(range(f_start, f_end)):
            X_col = chunk[:, i].astype(np.float64)
            pct = np.percentile(X_col, np.linspace(0, 100, 50))
            mean_pos = float(X_col[pos_mask].mean())
            mean_neg = float(X_col[neg_mask].mean())
            thresholds = np.unique(np.concatenate([pct, [mean_pos, mean_neg, (mean_pos+mean_neg)/2]]))

            X_exp = X_col[:, None]
            thr_ex = thresholds[None, :]
            y_exp = y_bool[:, None]
            w_exp = w[:, None]

            preds_pos = (X_exp < thr_ex)
            err_pos = (w_exp * (preds_pos != y_exp)).sum(axis=0)
            err_neg = (w_exp * (~preds_pos != y_exp)).sum(axis=0)

            i_pos, i_neg = np.argmin(err_pos), np.argmin(err_neg)
            if err_pos[i_pos] <= err_neg[i_neg]:
                err, thr, pol = float(err_pos[i_pos]), float(thresholds[i_pos]), 1
            else:
                err, thr, pol = float(err_neg[i_neg]), float(thresholds[i_neg]), -1

            if err < best_err:
                best_err, best_f, best_thr, best_pol = err, f_idx, thr, pol

        return best_err, best_f, best_thr, best_pol

    n_jobs = os.cpu_count()
    chunk_size = max(500, n_features // (n_jobs * 4))
    chunks = [(i, min(i + chunk_size, n_features)) for i in range(0, n_features, chunk_size)]

    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(eval_chunk)(f_start, f_end) for f_start, f_end in chunks
    )

    best_err, best_f, best_thr, best_pol = min(results, key=lambda r: r[0])

    best_err = float(np.clip(best_err, 1e-10, 1.0 - 1e-10))
    clf   = WeakClassifier(best_f, best_thr, best_pol)
    beta  = best_err / (1.0 - best_err)
    alpha = float(np.log(1.0 / beta))

    col   = np.array(X_ram[:, best_f], dtype=np.int64)
    preds = (best_pol * col < best_pol * best_thr).astype(np.int32)
    correct = (preds == np.asarray(y, dtype=np.int32))

    new_weights = w * np.where(correct, beta, 1.0)
    new_weights /= new_weights.sum()

    print(f'    -> clf idx={best_f} | err={best_err:.4f} | α={alpha:.4f} | {time.time()-t0:.1f}s', flush=True)
    return clf, alpha, new_weights, best_err

def strong_classifier_scores(X, classifiers, alphas):
    if len(classifiers) == 0:
        return np.zeros(X.shape[0], dtype=np.float64)
    preds_matrix = np.stack([clf.predict(X) for clf in classifiers], axis=0).astype(np.float64)
    alphas_arr   = np.array(alphas, dtype=np.float64)
    return (alphas_arr[:, None] * preds_matrix).sum(axis=0)

def find_theta(scores_pos, target_detection):
    thresholds = np.sort(scores_pos)[::-1]
    for theta in thresholds:
        if np.mean(scores_pos >= theta) >= target_detection:
            return float(theta)
    return float(thresholds[-1])


# ──────────────────────────────────────────────────────────────
# 6. PREDICCIÓN Y CHECKPOINTING
# ──────────────────────────────────────────────────────────────

def cascade_predict_online(img, cascade):
    img = img.astype(np.float64)
    h,w = img.shape
    ii = integral_image(img)
    ii2 = integral_image(img**2)
    std = normalize_variance(ii, ii2, h, w)
    if std <= 0:
        std = 1.0
    for stage in cascade:
        score = 0.0
        for clf, alpha in zip(stage['classifiers'], stage['alphas']):
            x, y, w, h, ft = stage['features_global'][clf.feature_index]
            val = compute_feature(ii, x, y, w, h, ft)
            val = val / std
            if clf.polarity * val < clf.polarity * clf.threshold:
                score += alpha
        if score < stage['theta']:
            return 0
    return 1

def _stage_ckpt_path(stage_idx): return os.path.join(CHECKPOINT_DIR, f'stage_{stage_idx+1:02d}_complete.pkl')
def _intra_stage_ckpt_path(stage_idx): return os.path.join(CHECKPOINT_DIR, f'stage_{stage_idx+1:02d}_inprogress.pkl')

def save_intra_stage(stage_idx, clfs, alphas, weights, mmap_path, mmap_shape, y_stage, n_pos, current_fpr, features):
    # NOTA: 'features' NO se guarda en pkl para ahorrar memoria.
    # Se regenera en memoria con generate_features() (es determinista).
    data = {
        'stage_idx':  stage_idx,
        'classifiers': clfs,
        'alphas':      alphas,
        'weights':     weights,
        'mmap_path':   mmap_path,
        'mmap_shape':  mmap_shape,
        'y_stage':     y_stage,
        'n_pos':       n_pos,
        'current_fpr': current_fpr,
    }
    path = _intra_stage_ckpt_path(stage_idx)
    with open(path + '.tmp', 'wb') as f:
        pickle.dump(data, f, protocol=4)
    os.replace(path + '.tmp', path)

def load_cascade_state():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cascade, stage_idx = [], 0
    while os.path.exists(_stage_ckpt_path(stage_idx)):
        with open(_stage_ckpt_path(stage_idx), 'rb') as f:
            cascade.append(pickle.load(f))
        print(f'  ✅ Etapa {stage_idx+1} cargada desde checkpoint.')
        stage_idx += 1
    inprogress = None
    ip_path = _intra_stage_ckpt_path(stage_idx)
    if os.path.exists(ip_path):
        with open(ip_path, 'rb') as f:
            inprogress = pickle.load(f)
        print(f'  🔄 Etapa {stage_idx+1} en progreso ({len(inprogress["classifiers"])} clfs).')
    return cascade, stage_idx, inprogress


# ──────────────────────────────────────────────────────────────
# 7. ENTRENAMIENTO DE CASCADA
# ──────────────────────────────────────────────────────────────

def train_cascade(X_pos_pool, backgrounds_pool, num_stages=6, target_detection=0.995,
                  target_fpr=0.50, max_features=None, save_every_n_clfs=5, n_jobs=-1,
                  X_pos_test=None, X_neg_test=None, img_shape=(24, 24)):
    """
    X_pos_test : lista de imágenes positivas de test (cargadas desde disco, estáticas).
    X_neg_test : lista de crops negativos de test (cargados desde disco, estáticos).
    img_shape  : shape de las imágenes (para generate_features).
    """

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cascade, start_stage, inprogress = load_cascade_state()

    if start_stage >= num_stages:
        print('Cascada ya completamente entrenada.')
        return cascade

    # ── Evaluar stages ya cargados desde checkpoint ────────
    if X_pos_test is not None and X_neg_test is not None and len(cascade) > 0:
        print('\nEvaluando stages cargados desde checkpoint...')
        for i in range(1, len(cascade) + 1):
            cascade_parcial = cascade[:i]
            preds_p = np.array(
                [cascade_predict_online(img, cascade_parcial) for img in X_pos_test],
                dtype=np.int32
            )
            preds_n = np.array(
                [cascade_predict_online(img, cascade_parcial) for img in X_neg_test],
                dtype=np.int32
            )
            dr_t  = float(np.mean(preds_p == 1))
            fpr_t = float(np.mean(preds_n == 1))
            print(f'  [TEST ckpt] Etapa {i:2d} | DR={dr_t*100:.2f}%  FPR={fpr_t*100:.4f}%')
    # ───────────────────────────────────────────────────────

    # Las features se generan en memoria (NO se leen/escriben a pkl).
    features = generate_features(img_shape, step=1, max_features=max_features)

    hard_neg_pool = []

    for stage_idx in range(start_stage, num_stages):
        print(f'\n{"="*55}\n  ETAPA {stage_idx+1}/{num_stages}\n{"="*55}', flush=True)

        if inprogress is not None and inprogress['stage_idx'] == stage_idx:
            stage_clfs   = inprogress['classifiers']
            stage_alphas = inprogress['alphas']
            y_stage      = inprogress['y_stage']
            n_pos        = inprogress['n_pos']
            mmap_path    = inprogress['mmap_path']
            mmap_shape   = inprogress['mmap_shape']
            X_feats      = np.memmap(mmap_path, dtype=np.float32, mode='r', shape=mmap_shape)
            inprogress   = None

            print(f'  Recalculando pesos...', flush=True)
            weights = np.ones(mmap_shape[0], dtype=np.float64) / mmap_shape[0]
            for clf, alpha in zip(stage_clfs, stage_alphas):
                beta    = float(np.exp(-alpha))
                col     = np.array(X_feats[:, clf.feature_index], dtype=np.int64)
                preds   = (clf.polarity * col < clf.polarity * clf.threshold).astype(np.int32)
                weights = weights * np.where(
                    preds == np.asarray(y_stage, dtype=np.int32), beta, 1.0
                )
                weights /= weights.sum()

            scores_pos  = strong_classifier_scores(X_feats[:n_pos], stage_clfs, stage_alphas)
            scores_neg  = strong_classifier_scores(X_feats[n_pos:], stage_clfs, stage_alphas)
            theta_i     = find_theta(scores_pos, target_detection)
            current_fpr = float(np.mean(scores_neg >= theta_i))
            det_rate    = float(np.mean(scores_pos >= theta_i))

        else:
            n_pos = len(X_pos_pool)
            needed = n_pos
            X_neg_current = []

            if len(hard_neg_pool) > 0:
                X_neg_current = hard_neg_pool[:needed]
                hard_neg_pool = hard_neg_pool[needed:]

            if len(X_neg_current) < needed:
                extra_needed = needed - len(X_neg_current)
                new_neg = _mine_dynamic_negatives(
                    backgrounds_pool,
                    cascade,
                    extra_needed
                )
                X_neg_current.extend(new_neg)

            n_neg = len(X_neg_current)
            if n_neg == 0:
                print('No se pudieron extraer suficientes negativos. Fin del entrenamiento.')
                break

            n_total = n_pos + n_neg
            y_stage = np.array([1]*n_pos + [0]*n_neg, dtype=np.int32)

            mmap_path  = os.path.join(CHECKPOINT_DIR, f'stage_feats.dat')
            mmap_shape = (n_total, len(features))
            print(f'  Extrayendo features → {mmap_path}', flush=True)

            mmap = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=mmap_shape)
            del mmap

            _fill_mmap_chunk(X_pos_pool,    features, mmap_path, mmap_shape, 0,     True,  n_jobs, 'positivos')
            _fill_mmap_chunk(X_neg_current, features, mmap_path, mmap_shape, n_pos, False, n_jobs, 'negativos')

            X_feats = np.memmap(mmap_path, dtype=np.float32, mode='r', shape=mmap_shape)

            w_pos, w_neg = 1.0 / (2 * n_pos), 1.0 / (2 * n_neg)
            weights = np.array([w_pos]*n_pos + [w_neg]*n_neg, dtype=np.float64)
            weights /= weights.sum()

            stage_clfs, stage_alphas, theta_i, current_fpr, det_rate = [], [], 0.0, 1.0, 0.0

        # ── Bucle AdaBoost ─────────────────────────────────────
        while current_fpr > target_fpr:
            print(f'\n  -- Ronda AdaBoost #{len(stage_clfs)+1} --', flush=True)
            clf, alpha, weights, err = adaboost_round(
                X_feats, y_stage, weights, mmap_path, mmap_shape
            )
            stage_clfs.append(clf)
            stage_alphas.append(alpha)

            scores_pos  = strong_classifier_scores(X_feats[:n_pos], stage_clfs, stage_alphas)
            scores_neg  = strong_classifier_scores(X_feats[n_pos:], stage_clfs, stage_alphas)
            theta_i     = find_theta(scores_pos, target_detection)
            current_fpr = float(np.mean(scores_neg >= theta_i))
            det_rate    = float(np.mean(scores_pos >= theta_i))

            print(
                f'  Clf {len(stage_clfs):3d} | FPR={current_fpr:.4f} | DR={det_rate:.4f} | '
                f'err={err:.4f} | θ={theta_i:.4f}',
                flush=True
            )

            if len(stage_clfs) % save_every_n_clfs == 0:
                save_intra_stage(
                    stage_idx, stage_clfs, stage_alphas, weights,
                    mmap_path, mmap_shape, y_stage, n_pos, current_fpr,
                    features  # se pasa pero NO se serializa en el pkl
                )

            if len(stage_clfs) >= 200:
                print('  ⚠️  Límite de 200 clfs alcanzado.')
                break

        # ── Guardar etapa completa ──────────────────────────────
        stage_data = {
            'classifiers':     stage_clfs,
            'alphas':          stage_alphas,
            'theta':           theta_i,
            'features_global': features,   # sí se guarda en el stage pkl (es el modelo)
        }
        cascade.append(stage_data)

        # ── Evaluación sobre test estático ──────────────────────
        if X_pos_test is not None and X_neg_test is not None:
            preds_p = np.array(
                [cascade_predict_online(img, cascade) for img in X_pos_test],
                dtype=np.int32
            )
            preds_n = np.array(
                [cascade_predict_online(img, cascade) for img in X_neg_test],
                dtype=np.int32
            )
            dr_t  = float(np.mean(preds_p == 1))
            fpr_t = float(np.mean(preds_n == 1))
            print(f'  *** [TEST] DR={dr_t*100:.2f}%  FPR={fpr_t*100:.4f}% ***', flush=True)
        # ───────────────────────────────────────────────────────

        print("  Guardando hard negatives para siguiente etapa...")
        remaining_hard = [img for img in X_neg_current if cascade_predict_online(img, cascade) == 1]
        hard_neg_pool.extend(remaining_hard)

        with open(_stage_ckpt_path(stage_idx) + '.tmp', 'wb') as f:
            pickle.dump(stage_data, f, protocol=4)
        os.replace(_stage_ckpt_path(stage_idx) + '.tmp', _stage_ckpt_path(stage_idx))
        if os.path.exists(_intra_stage_ckpt_path(stage_idx)):
            os.remove(_intra_stage_ckpt_path(stage_idx))

    return cascade


# ──────────────────────────────────────────────────────────────
# 8. EVALUACIÓN Y MAIN
# ──────────────────────────────────────────────────────────────

def evaluate(X_eval, y_eval, cascade, desc='', n_jobs=-1):
    preds = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(cascade_predict_online)(img, cascade) for img in tqdm(X_eval, desc=f'Ev. {desc}')
    )
    preds = np.array(preds, dtype=np.int32)
    tn, fp, fn, tp = confusion_matrix(y_eval, preds, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f'\n─ RESULTADOS: {desc} ─')
    print(f'  TP: {tp:6d}  |  FP: {fp:6d}\n  FN: {fn:6d}  |  TN: {tn:6d}')
    print(f'  Precisión : {precision*100:.2f}%\n  Recall(DR): {recall*100:.2f}%')
    print(f'  FPR       : {fpr*100:.4f}%\n  F1-score  : {f1*100:.2f}%')
    return preds


if __name__ == '__main__':
    # ── Rutas ──────────────────────────────────────────────────
    CHECKPOINT_DIR       = 'checkpoints_total'
    PATH_FACES_TRAIN     = 'dataset/faces_clean'        # Caras de entrenamiento
    PATH_FACES_TEST      = 'dataset/faces_test'         # Caras de test (carpeta fija, estática)
    PATH_BACKGROUNDS     = 'dataset/background_clean'           # Fondos para mining de negativos (train)
    PATH_BACKGROUNDS_TEST= 'dataset/background_crops'            # Fondos para crops negativos de test

    # ── Hiperparámetros ────────────────────────────────────────
    NUM_STAGES   = 20
    TARGET_DR    = 0.99
    TARGET_FPR   = 0.50
    MAX_FEATURES = None
    SAVE_EVERY   = 5
    N_JOBS       = -1
    NUM_NEG_TEST = 100_000 

    # ── Carga de datos de entrenamiento ────────────────────────
    X_pos_train  = load_images(PATH_FACES_TRAIN)
    bg_train     = load_background_paths(PATH_BACKGROUNDS)

    # ── Carga de datos de test (estáticos, desde disco) ────────
    print('\nCargando imágenes de test positivas...')
    X_pos_test = load_images(PATH_FACES_TEST, test=True)

    print(f'\nMinando {NUM_NEG_TEST} crops negativos de test (operación única)...')
    X_neg_test    = load_images(PATH_BACKGROUNDS_TEST, is_negative=True, test=True)

    print(f'\nTrain positivos : {len(X_pos_train)}')
    print(f'Train negativos minados en cada ronda: {len(X_pos_train)}')
    print(f'Test  positivos : {len(X_pos_test)}')
    print(f'Test  negativos : {len(X_neg_test)}\n')

    # ── Entrenamiento ──────────────────────────────────────────
    cascade = train_cascade(
        X_pos_pool        = X_pos_train,
        backgrounds_pool  = bg_train,
        num_stages        = NUM_STAGES,
        target_detection  = TARGET_DR,
        target_fpr        = TARGET_FPR,
        max_features      = MAX_FEATURES,
        save_every_n_clfs = SAVE_EVERY,
        n_jobs            = N_JOBS,
        X_pos_test        = X_pos_test,
        X_neg_test        = X_neg_test,
        img_shape         = (24, 24),
    )

    # ── Evaluación final ───────────────────────────────────────
    y_pos_test = np.ones (len(X_pos_test), dtype=np.int32)
    y_neg_test = np.zeros(len(X_neg_test), dtype=np.int32)

    evaluate(X_pos_test, y_pos_test, cascade, desc='POSITIVOS', n_jobs=N_JOBS)
    evaluate(X_neg_test, y_neg_test, cascade, desc='NEGATIVOS', n_jobs=N_JOBS)

    with open(os.path.join(CHECKPOINT_DIR, 'cascade_final.pkl'), 'wb') as f:
        pickle.dump(cascade, f, protocol=4)
    print('\n✅ Modelo final guardado en cascade_final.pkl')