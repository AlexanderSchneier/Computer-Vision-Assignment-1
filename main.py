import os
import cv2
import numpy as np
import scipy.io
from scipy.ndimage import convolve
from scipy.io import loadmat
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel


#PART A -- 1 and 2
IMG_DIR = "images" 
FILTERS_MAT = "filters/filters.mat"
OUT_DIR = "Images_with_filters"
SIZE = (100, 100) 
os.makedirs(OUT_DIR, exist_ok=True)

F = loadmat(FILTERS_MAT)['F']

#PART A -- 3
def load_gray_resized_and_convolve(path, F ,size=SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Could not read image at {path}")
        return

    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,size).astype(np.float32)
    
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(OUT_DIR, base)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(F.shape[2]):
        kernel = F[:, :, i].astype(np.float32)
        individual_images = convolve(gray, kernel, mode='reflect')
        vis = cv2.normalize(individual_images, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        out_path = os.path.join(out_dir,f"{base}_filter{i}.png")
        cv2.imwrite(out_path, vis)

for fname in os.listdir(IMG_DIR):
    path = os.path.join(IMG_DIR, fname)
    if not os.path.isfile(path): 
        continue
    load_gray_resized_and_convolve(path, F, SIZE)

#part 4
categories = {
    "cardinal": ["cardinal1.jpg", "cardinal2.jpg"],
    "leopard":  ["leopard1.jpg",  "leopard2.jpg"],
    "panda":    ["panda1.jpg",    "panda2.jpg"],
}

grids_dir = os.path.join(OUT_DIR, "grids")
os.makedirs(grids_dir, exist_ok=True)

def show_filter_grid(filter_idx):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 12))

    kernel = F[:, :, filter_idx]
    m = np.abs(kernel).max() or 1.0
    axes[0, 0].imshow(kernel, cmap="gray", vmin=-m, vmax=m)
    axes[0, 0].set_title(f"Filter {filter_idx}")
    axes[0, 0].axis("off")

    axes[0, 1].axis("off")

    row = 1
    for animal, files in categories.items():
        for col, fname in enumerate(files):
            base = os.path.splitext(fname)[0]
            resp_path = os.path.join(OUT_DIR, base, f"{base}_filter{filter_idx}.png")
            img = mpimg.imread(resp_path)
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(f"{animal}: {fname}")
            axes[row, col].axis("off")
        row += 1

    plt.tight_layout()
    save_path = os.path.join(grids_dir, f"filter_{filter_idx:02d}_grid.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

for i in range(F.shape[2]):
    show_filter_grid(i)


#part B

def computeTextureReprs(img, F, size=SIZE):
    if isinstance(img, str):
        arr = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Could not read image at {img}")
    else:
        arr = img

    if arr.ndim == 2:
        gray = arr
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, size).astype(np.float32) / 255.0

    num_filters = F.shape[2]
    h, w = gray.shape
    responses = np.zeros((num_filters, h, w), dtype=np.float32)

    for i in range(num_filters):
        responses[i] = convolve(gray, F[:, :, i].astype(np.float32), mode='reflect')

    texture_repr_concat = responses.ravel()                # (num_filters*h*w,)
    texture_repr_mean   = responses.mean(axis=(1, 2))      # (num_filters,)
    return texture_repr_concat, texture_repr_mean

#part C

im1_path = "images/baby_happy.jpg"
im2_path = "images/baby_weird.jpg"

OUT_DIR = "submission_images"
os.makedirs(OUT_DIR, exist_ok=True)

SIZE = (512, 512)
SIGMA = 6.0

def read_gray_resized(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def to_float01(img_u8):
    return img_u8.astype(np.float32) / 255.0

def to_u8(img_f):
    return (np.clip(img_f, 0.0, 1.0) * 255.0).astype(np.uint8)

im1 = read_gray_resized(im1_path, SIZE)
im2 = read_gray_resized(im2_path, SIZE)

im1f = to_float01(im1)
im2f = to_float01(im2)

im1_blur = gaussian_filter(im1f, sigma=SIGMA)
im2_blur = gaussian_filter(im2f, sigma=SIGMA)

cv2.imwrite(os.path.join(OUT_DIR, "im1_blur.png"), to_u8(im1_blur))
cv2.imwrite(os.path.join(OUT_DIR, "im2_blur.png"), to_u8(im2_blur))

im2_detail = im2f - im2_blur
detail_vis = (im2_detail - im2_detail.min()) / (im2_detail.max() - im2_detail.min() + 1e-8)
cv2.imwrite(os.path.join(OUT_DIR, "im2_detail.png"), to_u8(detail_vis))

hybrid = im1_blur + im2_detail
cv2.imwrite(os.path.join(OUT_DIR, "hybrid.png"), to_u8(hybrid))


#part D

OUT_DIR = "submission_images"
os.makedirs(OUT_DIR, exist_ok=True)

butterfly_path = "images/butterfly.jpg"
bf = read_gray_resized(butterfly_path, (256, 256))  # <- reuse Part C function
bf = bf.astype(np.float32) / 255.0

# Smooth with Gaussian
SIGMA = 1.2
bf_smooth = gaussian_filter(bf, sigma=SIGMA)

# Sobel gradients
Ix = sobel(bf_smooth, axis=1)  # x-gradient
Iy = sobel(bf_smooth, axis=0)  # y-gradient

# Save Ix, Iy
def to_u8(x):
    x = x - x.min()
    if x.max() > 0:
        x = x / (x.max() + 1e-8)
    return (255.0 * x).astype(np.uint8)

cv2.imwrite(os.path.join(OUT_DIR, "Ix.png"), to_u8(np.abs(Ix)))
cv2.imwrite(os.path.join(OUT_DIR, "Iy.png"), to_u8(np.abs(Iy)))

grad_img = np.hypot(Ix, Iy)
cv2.imwrite(os.path.join(OUT_DIR, "grad_img.jpg"), to_u8(grad_img))

theta = np.degrees(np.arctan2(Iy, Ix))
theta = (theta + 180.0) % 180.0

q_theta = np.zeros_like(theta, dtype=np.uint8)
q_theta[((theta >= 22.5) & (theta < 67.5))] = 1
q_theta[((theta >= 67.5) & (theta < 112.5))] = 2
q_theta[((theta >= 112.5) & (theta < 157.5))] = 3

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
        if g <= 0: continue
        b = int(q_theta[i, j])
        (di1, dj1), (di2, dj2) = dir_offsets[b]
        g1 = grad_img[i+di1, j+dj1]
        g2 = grad_img[i+di2, j+dj2]
        if g >= g1 and g >= g2:
            nms[i, j] = g

cv2.imwrite(os.path.join(OUT_DIR, "non_maxima_supp.jpg"), to_u8(nms))

mag_max = nms.max() + 1e-8
T2 = 0.25 * mag_max
T1 = 0.10 * mag_max

strong = (nms >= T2)
weak = ((nms >= T1) & ~strong)
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
                if edges[i+di1, j+dj1] or edges[i+di2, j+dj2]:
                    new_edges[i, j] = True
                    changed = True
    edges = new_edges

butterfly_edges = np.zeros_like(nms, dtype=np.uint8)
butterfly_edges[edges] = 255
cv2.imwrite(os.path.join(OUT_DIR, "butterfly_edges.png"), butterfly_edges)


#part e

def extract_keypoints(image_bgr, k=0.05, window_size=5, use_avg_times=5.0, top_percent=None):
    assert window_size % 2 == 1, "window_size must be odd"
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

    ys_all, xs_all = np.where(R > 0)
    if xs_all.size == 0:
        return np.array([]), np.array([]), np.array([]), Ix, Iy

    if top_percent is not None and 0 < top_percent <= 100:
        flat_inds = np.argsort(R[ys_all, xs_all])[::-1]
        n_keep = max(1, int(round(xs_all.size * (top_percent / 100.0))))
        keep_inds = flat_inds[:n_keep]
        thr_mask = np.zeros_like(R, dtype=bool)
        thr_mask[ys_all[keep_inds], xs_all[keep_inds]] = True
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
            if center > np.max(neighbors):  # strict
                nms_mask[i, j] = True

    ys, xs = np.where(nms_mask)
    sc = R[ys, xs]
    if xs.size == 0:
        return np.array([]), np.array([]), np.array([]), Ix, Iy

    x = xs.reshape(-1, 1).astype(np.int32)
    y = ys.reshape(-1, 1).astype(np.int32)
    scores = sc.reshape(-1, 1).astype(np.float32)
    return x, y, scores, Ix, Iy

def visualize_keypoints(img_bgr, x, y, scores, out_path):
    vis = img_bgr.copy()
    if scores.size > 0:
        s = scores.flatten()
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)  # 0..1
        for cx, cy, w in zip(x.flatten(), y.flatten(), s):
            r = int(2 + 6 * w)   # radius 2..8
            cv2.circle(vis, (int(cx), int(cy)), r, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(out_path, vis)

for fname in ["cardinal1.jpg", "leopard1.jpg", "panda1.jpg"]:
    path = os.path.join("images", fname)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Missing {path}, skipping.")
        continue

    x, y, scores, Ix_H, Iy_H = extract_keypoints(
        img,
        k=0.05,
        window_size=5,
        use_avg_times=5.0,
        top_percent=None 
    )

    out_png = os.path.join(OUT_DIR, f"{os.path.splitext(fname)[0]}.png")
    visualize_keypoints(img, x, y, scores, out_png)

