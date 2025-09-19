import cv2
import os
import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt

in_dir = 'images'
SIZE = (1000,1000)
out_dir = 'images_with_filters'
single_img = 'images/baby_happy.jpg'
filters = loadmat('filters/filters.mat')

def resize(in_directory , out_directory, size = (1000,1000)):
    for filename in os.listdir(in_directory):
        path = os.path.join(in_directory, filename)
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        img_resized = cv2.resize(img, size)
        out_path = os.path.join(out_directory, filename)
        cv2.imwrite(out_path, img_resized)

def black_and_white(in_directory, out_directory):
    for filename in os.listdir(in_directory):
        path = os.path.join(in_directory, filename)
        img = cv2.imread(path)

        img_black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out_path = os.path.join(out_directory, filename)
        cv2.imwrite(out_path, img_black_and_white)

def convolve(filters, image):
    for filter in filters:




resize(in_dir, out_dir)
black_and_white(out_dir, out_dir)

