import os
import cv2
import numpy as np
import scipy.io
from scipy.ndimage import convolve
from scipy.io import loadmat
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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

for images in os.listdir(IMG_DIR):
    load_gray_resized_and_convolve(os.path.join(IMG_DIR,images), F, SIZE)


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

    # --- Row 1, Col 1: visualize the filter kernel ---
    kernel = F[:, :, filter_idx]
    m = np.abs(kernel).max() or 1.0
    axes[0, 0].imshow(kernel, cmap="gray", vmin=-m, vmax=m)
    axes[0, 0].set_title(f"Filter {filter_idx}")
    axes[0, 0].axis("off")

    # --- Row 1, Col 2: blank subplot ---
    axes[0, 1].axis("off")

    # --- Rows 2–4: responses for each animal pair ---
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

# build a grid image for every filter
for i in range(F.shape[2]):
    show_filter_grid(i)


