import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

IMG = 'images'
SIZE = (100,100)
out_dir = 'images_with_filters'

for filename in os.listdir(IMG):
    path = os.path.join(IMG, filename)
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, SIZE)
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, img_resized)

