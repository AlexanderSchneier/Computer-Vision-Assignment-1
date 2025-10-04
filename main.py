import os
import glob
import numpy as np
import cv2
from scipy.io import loadmat
from scipy.ndimage import convolve, gaussian_filter, sobel

IMG_DIR       = "images"
FILTERS_MAT   = "filters/filters.mat"
OUT_A_DIR     = "Images_with_filters" 
SUBMIT_DIR    = "submission_images" 
os.makedirs(OUT_A_DIR, exist_ok=True)
os.makedirs(SUBMIT_DIR, exist_ok=True)

categories = {
    "cardinal": ["cardinal1.jpg", "cardinal2.jpg"],
    "leopard":  ["leopard1.jpg",  "leopard2.jpg"],
    "panda":    ["panda1.jpg",    "panda2.jpg"],
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def read_gray_resized(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def read_color_resized(path, size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def to_float01(img_u8):
    return img_u8.astype(np.float32) / 255.0

def to_u8(img_f):
    return (np.clip(img_f, 0.0, 1.0) * 255.0).astype(np.uint8)

def euclid(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b))

# Part A 
def load_filters(FILTERS_MAT):
    F = loadmat(FILTERS_MAT)["F"]
    return F.astype(np.float32)

def load_gray_resized_and_convolve(path, F, size=(100, 100)):
    ext = os.path.splitext(path)[1].lower()
    if ext not in VALID_EXTS:
        return
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[Part A] WARN: Could not read {path}")
        return
    # Ensure grayscale
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print(f"[Part A] WARN: Unexpected shape {img.shape} for {path}")
        return

    gray = cv2.resize(gray, size).astype(np.float32) / 255.0

    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(OUT_A_DIR, base)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(F.shape[2]):
        kernel = F[:, :, i].astype(np.float32)
        resp = convolve(gray, kernel, mode='reflect')
        vis = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{base}_filter{i}.png"), vis)

def run_part_a(F):
    print("[Part A] Generating per-filter response images...")
    for fname in os.listdir(IMG_DIR):
        path = os.path.join(IMG_DIR, fname)
        if not os.path.isfile(path):
            continue
        load_gray_resized_and_convolve(path, F, size=(100, 100))
    print(f"[Part A] Saved responses under {OUT_A_DIR}/")

# Part B 
def computeTextureReprs(img_or_path, F, size=(100, 100)):
    """Returns (texture_repr_concat, texture_repr_mean)."""
    if isinstance(img_or_path, str):
        arr = cv2.imread(img_or_path, cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"Could not read {img_or_path}")
    else:
        arr = img_or_path
    gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size).astype(np.float32) / 255.0

    num_filters = F.shape[2]
    h, w = gray.shape
    responses = np.zeros((num_filters, h, w), dtype=np.float32)
    for i in range(num_filters):
        responses[i] = convolve(gray, F[:, :, i].astype(np.float32), mode='reflect')

    texture_repr_concat = responses.ravel().astype(np.float32)
    texture_repr_mean   = responses.mean(axis=(1, 2)).astype(np.float32)
    return texture_repr_concat, texture_repr_mean

# Part C
def run_part_c():
    print("[Part C] Building hybrid image...")
    im1_path = os.path.join(IMG_DIR, "woman_happy.png")
    im2_path = os.path.join(IMG_DIR, "woman_neutral.png")
    SIZE = (512, 512)
    SIGMA = 6.0
    im1 = read_gray_resized(im1_path, SIZE).astype(np.float32) / 255.0
    im2 = read_gray_resized(im2_path, SIZE).astype(np.float32) / 255.0
    im1_blur = gaussian_filter(im1, sigma=SIGMA)
    im2_blur = gaussian_filter(im2, sigma=SIGMA)
    cv2.imwrite(os.path.join(SUBMIT_DIR, "im1_blur.png"), to_u8(im1_blur))
    cv2.imwrite(os.path.join(SUBMIT_DIR, "im2_blur.png"), to_u8(im2_blur))
    im2_detail = im2 - im2_blur
    detail_vis = (im2_detail - im2_detail.min()) / (im2_detail.max() - im2_detail.min() + 1e-8)
    cv2.imwrite(os.path.join(SUBMIT_DIR, "im2_detail.png"), to_u8(detail_vis))
    hybrid = im1_blur + im2_detail
    cv2.imwrite(os.path.join(SUBMIT_DIR, "hybrid.png"), to_u8(hybrid))
    print(f"[Part C] Saved hybrid outputs under {SUBMIT_DIR}/")

# Part D
def run_part_d():
    print("[Part D] Running edge detection pipeline...")
    SIZE = (256, 256)
    bf_path = os.path.join(IMG_DIR, "butterfly.jpg")
    bf = read_gray_resized(bf_path, SIZE).astype(np.float32) / 255.0

    SIGMA = 1.2
    sm = gaussian_filter(bf, sigma=SIGMA)

    Ix = sobel(sm, axis=1)
    Iy = sobel(sm, axis=0)
    cv2.imwrite(os.path.join(SUBMIT_DIR, "Ix.png"), to_u8(np.abs(Ix)))
    cv2.imwrite(os.path.join(SUBMIT_DIR, "Iy.png"), to_u8(np.abs(Iy)))

    grad_img = np.hypot(Ix, Iy)
    cv2.imwrite(os.path.join(SUBMIT_DIR, "grad_img.jpg"), to_u8(grad_img))

    theta = np.degrees(np.arctan2(Iy, Ix))
    theta = (theta + 180.0) % 180.0
    q_theta = np.zeros_like(theta, dtype=np.uint8)
    q_theta[(theta >= 22.5)  & (theta < 67.5)]   = 1
    q_theta[(theta >= 67.5)  & (theta < 112.5)]  = 2
    q_theta[(theta >= 112.5) & (theta < 157.5)]  = 3

    H, W = grad_img.shape
    nms = np.zeros_like(grad_img, dtype=np.float32)
    dir_offsets = {
        0:   ((0, -1), (0,  1)),
        1:   ((-1, 1), (1, -1)),
        2:   ((-1, 0), (1,  0)),
        3:   ((-1,-1), (1,  1)),
    }
    for i in range(1, H-1):
        for j in range(1, W-1):
            g = grad_img[i, j]
            if g <= 0: 
                continue
            b = int(q_theta[i, j])
            (di1, dj1), (di2, dj2) = dir_offsets[b]
            g1 = grad_img[i + di1, j + dj1]
            g2 = grad_img[i + di2, j + dj2]
            if g >= g1 and g >= g2:
                nms[i, j] = g
    cv2.imwrite(os.path.join(SUBMIT_DIR, "non_maxima_supp.jpg"), to_u8(nms))

    mag_max = nms.max() + 1e-8
    T2 = 0.25 * mag_max
    T1 = 0.10 * mag_max
    strong = (nms >= T2)
    weak   = ((nms >= T1) & ~strong)
    edges = strong.copy()

    changed = True
    while changed:
        changed = False
        new_edges = edges.copy()
        for i in range(1, H-1):
            for j in range(1, W-1):
                if weak[i, j] and not edges[i, j]:
                    b = int(q_theta[i, j])
                    (di1, dj1), (di2, dj2) = dir_offsets[b]
                    if edges[i + di1, j + dj1] or edges[i + di2, j + dj2]:
                        new_edges[i, j] = True
                        changed = True
        edges = new_edges

    out = np.zeros_like(nms, dtype=np.uint8)
    out[edges] = 255
    cv2.imwrite(os.path.join(SUBMIT_DIR, "butterfly_edges.png"), out)
    print(f"[Part D] Saved edge outputs under {SUBMIT_DIR}/")

# Part E — Harris corner detector + visualization
def extract_keypoints(image_bgr, k=0.05, window_size=5, use_avg_times=5.0, top_percent=None):
    assert window_size % 2 == 1
    half = window_size // 2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    H, W = gray.shape

    Ix = sobel(gray, axis=1).astype(np.float32)
    Iy = sobel(gray, axis=0).astype(np.float32)

    Ixx, Iyy, Ixy = Ix*Ix, Iy*Iy, Ix*Iy
    R = np.zeros_like(gray, dtype=np.float64)

    for i in range(half, H - half):
        for j in range(half, W - half):
            if i < 2 or j < 2 or i >= H-2 or j >= W-2:
                R[i, j] = 0.0
                continue
            win_Ixx = Ixx[i-half:i+half+1, j-half:j+half+1]
            win_Iyy = Iyy[i-half:i+half+1, j-half:j+half+1]
            win_Ixy = Ixy[i-half:i+half+1, j-half:j+half+1]
            Sxx = float(np.sum(win_Ixx))
            Syy = float(np.sum(win_Iyy))
            Sxy = float(np.sum(win_Ixy))
            detM   = (Sxx * Syy) - (Sxy * Sxy)
            traceM = (Sxx + Syy)
            R[i, j] = detM - k * (traceM ** 2)

    if top_percent is not None and 0 < top_percent <= 100:
        ys_all, xs_all = np.where(R > 0)
        if xs_all.size:
            flat_inds = np.argsort(R[ys_all, xs_all])[::-1]
            n_keep = max(1, int(round(xs_all.size * (top_percent / 100.0))))
            keep_inds = flat_inds[:n_keep]
            thr_mask = np.zeros_like(R, dtype=bool)
            thr_mask[ys_all[keep_inds], xs_all[keep_inds]] = True
        else:
            thr_mask = np.zeros_like(R, dtype=bool)
    else:
        thr = use_avg_times * float(np.mean(R[R > 0])) if np.any(R > 0) else 0.0
        thr_mask = (R > thr)

    nms_mask = np.zeros_like(thr_mask, dtype=bool)
    for i in range(1, H-1):
        for j in range(1, W-1):
            if not thr_mask[i, j]:
                continue
            patch = R[i-1:i+2, j-1:j+2]
            center = R[i, j]
            neighbors = np.delete(patch.reshape(-1), 4)
            if center > np.max(neighbors):
                nms_mask[i, j] = True

    ys, xs = np.where(nms_mask)
    sc = R[ys, xs]
    x = xs.reshape(-1, 1).astype(np.int32)
    y = ys.reshape(-1, 1).astype(np.int32)
    scores = sc.reshape(-1, 1).astype(np.float32)
    return x, y, scores, Ix, Iy

def visualize_keypoints(img_bgr, x, y, scores, out_path):
    vis = img_bgr.copy()
    if scores.size:
        s = scores.flatten()
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        for cx, cy, w in zip(x.flatten(), y.flatten(), s):
            r = int(2 + 6 * w)  # radius 2..8
            cv2.circle(vis, (int(cx), int(cy)), r, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(out_path, vis)

def run_part_e():
    print("[Part E] Running Harris corner detector on 3 images...")
    for fname in ["cardinal1.jpg", "leopard1.jpg", "panda1.jpg"]:
        path = os.path.join(IMG_DIR, fname)
        img = read_color_resized(path, (100, 100))  # any reasonable size is fine for viz
        x, y, scores, Ix_H, Iy_H = extract_keypoints(img, k=0.05, window_size=5, use_avg_times=5.0)
        out_png = os.path.join(SUBMIT_DIR, f"{os.path.splitext(fname)[0]}.png")
        visualize_keypoints(img, x, y, scores, out_png)
    print(f"[Part E] Saved visualizations under {SUBMIT_DIR}/")

# Part F
def compute_features(x, y, scores, Ix, Iy):
    H, W = Ix.shape
    x = np.asarray(x).flatten().astype(int)
    y = np.asarray(y).flatten().astype(int)
    n = x.size

    mag   = np.hypot(Ix, Iy)
    theta = np.degrees(np.arctan2(Iy, Ix))
    theta = ((theta + 90.0) % 180.0) - 90.0

    features = np.zeros((n, 8), dtype=np.float32)
    bin_width = 22.5

    for i in range(n):
        cx, cy = x[i], y[i]
        if cx < 5 or cy < 5 or cx > W - 6 or cy > H - 6:
            continue
        mpatch = mag  [cy-5:cy+6, cx-5:cx+6]
        apatch = theta[cy-5:cy+6, cx-5:cx+6]
        valid = mpatch > 0
        if not np.any(valid):
            continue
        m = mpatch[valid].astype(np.float32)
        a = apatch[valid]
        bins = np.floor((a + 90.0) / bin_width).astype(int)
        bins = np.clip(bins, 0, 7)
        hist = np.zeros(8, dtype=np.float32)
        for b in range(8):
            if np.any(bins == b):
                hist[b] = m[bins == b].sum(dtype=np.float32)
        hnorm = np.linalg.norm(hist) + 1e-8
        hist = hist / hnorm
        hist = np.minimum(hist, 0.2, dtype=np.float32)
        hnorm2 = np.linalg.norm(hist) + 1e-8
        hist = (hist / hnorm2).astype(np.float32)
        features[i, :] = hist
    return features

# Part G
def computeBOWRepr(features, means):
    features = np.asarray(features, dtype=np.float32)
    means    = np.asarray(means,    dtype=np.float32)
    if means.ndim != 2 or means.shape[1] != 8:
        raise ValueError("means must be (k, 8)")
    k = means.shape[0]
    if features.size == 0:
        return np.zeros(k, dtype=np.float32)
    dists = np.sum((features[:, None, :] - means[None, :, :])**2, axis=2)  # (n,k)
    closest = np.argmin(dists, axis=1)
    counts = np.bincount(closest, minlength=k).astype(np.float32)
    s = counts.sum()
    return counts / s if s > 0 else counts

def find_cluster_means_from_mat():
    candidates = [
        "mean.mat",
        os.path.join("images", "mean.mat"),
        os.path.join("filters", "mean.mat"),
        os.path.join("data", "mean.mat"),
        os.path.join("hw", "mean.mat"),
        os.path.join("homework", "mean.mat"),
    ]
    candidates += glob.glob("**/mean.mat", recursive=True)
    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand):
            try:
                mat = loadmat(cand)
                for key, val in mat.items():
                    if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == 8:
                        print(f"[Part G] Using cluster means from: {cand} (key='{key}', shape={val.shape})")
                        return val.astype(np.float32)
            except Exception as e:
                print(f"[Part G] Failed to read {cand}: {e}")
    return None

def build_cluster_means_via_kmeans(K=40, size=(100, 100)):
    try:
        from scipy.cluster.vq import kmeans2
    except Exception as e:
        raise RuntimeError("scipy.cluster.vq.kmeans2 not available; provide mean.mat") from e
    print(f"[Part G] mean.mat not found. Building K={K} vocab via k-means...")
    all_feats = []
    for cls, files in categories.items():
        for fname in files:
            path = os.path.join(IMG_DIR, fname)
            img_bgr = read_color_resized(path, size)
            x, y, scores, IxH, IyH = extract_keypoints(img_bgr, k=0.05, window_size=5, use_avg_times=5.0)
            feats = compute_features(x, y, scores, IxH, IyH)
            if feats.size:
                all_feats.append(feats.astype(np.float32))
    if not all_feats:
        raise RuntimeError("[Part G] No descriptors to build a vocabulary. Lower Harris threshold or check image paths.")
    X = np.vstack(all_feats).astype(np.float32)
    if X.shape[0] < K:
        K = max(2, min(int(X.shape[0]), K))
        print(f"[Part G] Not enough features; using K={K}.")
    centroids, _ = kmeans2(X, K, minit='++', iter=50)
    print(f"[Part G] Built cluster means: {centroids.shape} from {X.shape[0]} descriptors.")
    return centroids.astype(np.float32)

# Part H
def run_part_h(F):
    print("[Part H] Computing representation quality ratios...")

    cluster_means = find_cluster_means_from_mat()
    if cluster_means is None:
        cluster_means = build_cluster_means_via_kmeans(K=40, size=(100, 100))

    TEST_SIZE   = (100, 100)
    RESULTS_DOC = "results.doc"

    image_keys = []
    reps = {}
    for cls, files in categories.items():
        for fname in files:
            path = os.path.join(IMG_DIR, fname)
            tex_concat, tex_mean = computeTextureReprs(path, F, size=TEST_SIZE)
            img_bgr = read_color_resized(path, TEST_SIZE)
            x, y, scores, IxH, IyH = extract_keypoints(img_bgr, k=0.05, window_size=5, use_avg_times=5.0)
            feats = compute_features(x, y, scores, IxH, IyH)
            bow = computeBOWRepr(feats, cluster_means)
            key = f"{cls}/{fname}"
            reps[key] = {"class": cls, "t_concat": tex_concat, "t_mean": tex_mean, "bow": bow}
            image_keys.append(key)

    def compute_ratio(rep_name: str):
        within, between = [], []
        for i in range(len(image_keys)):
            for j in range(i + 1, len(image_keys)):
                a_key, b_key = image_keys[i], image_keys[j]
                a_cls, b_cls = reps[a_key]["class"], reps[b_key]["class"]
                d = euclid(reps[a_key][rep_name], reps[b_key][rep_name])
                (within if a_cls == b_cls else between).append(d)
        w = float(np.mean(within)) if within else float("nan")
        b = float(np.mean(between)) if between else float("nan")
        ratio = (w / b) if (np.isfinite(w) and np.isfinite(b) and b > 0) else float("nan")
        return w, b, ratio

    w_bow, b_bow, r_bow = compute_ratio("bow")
    w_tc,  b_tc,  r_tc  = compute_ratio("t_concat")
    w_tm,  b_tm,  r_tm  = compute_ratio("t_mean")

    print("[Part H] average_within / average_between ratios:")
    print(f"  BOW:                {r_bow:.6f}")
    print(f"  Texture (concat):   {r_tc:.6f}")
    print(f"  Texture (mean):     {r_tm:.6f}")

    def vec_preview(v, n=64):
        v = np.asarray(v).ravel()
        n = int(min(n, v.size))
        return np.array2string(v[:n], precision=6, separator=", ")

    def save_large_vector(v, base_name, save_dir=SUBMIT_DIR):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{base_name}.npy")
        np.save(path, np.asarray(v, dtype=np.float32))
        return path, vec_preview(v, n=64)

    with open(RESULTS_DOC, "w", encoding="utf-8") as f:
        f.write("Results for Part H: Representation Quality Test\n")
        f.write("================================================\n\n")
        f.write("Per-image representations:\n\n")
        for key in image_keys:
            cls = reps[key]["class"]
            bow = reps[key]["bow"]
            tmean = reps[key]["t_mean"]
            tconcat = reps[key]["t_concat"]

            base = key.replace("/", "_").replace(".jpg", "") + "_texture_concat"
            concat_path, concat_preview = save_large_vector(tconcat, base, save_dir=SUBMIT_DIR)

            f.write(f"Image: {key} (class={cls})\n")
            f.write(f"  bow_repr (k={bow.shape[0]}):\n    {np.array2string(bow, precision=6, separator=', ')}\n")
            f.write(f"  texture_repr_mean (len={tmean.shape[0]}):\n    {np.array2string(tmean, precision=6, separator=', ')}\n")
            f.write(f"  texture_repr_concat (len={tconcat.shape[0]}): saved to {concat_path}\n")
            f.write(f"    preview first 64: {concat_preview}\n\n")

        f.write("Average distance means and ratios (within / between):\n")
        f.write(f"  BOW:            within={w_bow:.6f}  between={b_bow:.6f}  ratio={r_bow:.6f}\n")
        f.write(f"  Texture concat: within={w_tc:.6f}   between={b_tc:.6f}   ratio={r_tc:.6f}\n")
        f.write(f"  Texture mean:   within={w_tm:.6f}   between={b_tm:.6f}   ratio={r_tm:.6f}\n")

    print(f"[Part H] Wrote {RESULTS_DOC} and saved large vectors under {SUBMIT_DIR}/")

# Main
if __name__ == "__main__":
    # Load filter bank
    F = load_filters(FILTERS_MAT)

    # Part A
    run_part_a(F)

    # Part C
    run_part_c()

    # Part D
    run_part_d()

    # Part E
    run_part_e()

    # Part H (uses Parts B, E, F, G functions)
    run_part_h(F)

    print("\nAll parts completed.")
