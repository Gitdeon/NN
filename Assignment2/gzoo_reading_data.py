import glob
import os
import numpy as np
import matplotlib as plt
import cv2 #pip3 install opencv-python

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID = 0

jpg_paths = glob.glob('data/images_training_rev1/*.jpg')
jpg_paths = np.sort(jpg_paths)

galaxy_images = []
for i in range(len(jpg_paths)):
    jpg = cv2.imread(jpg_paths[i])
    galaxy_images.append(cv2.resize(jpg, dsize=(128, 128), interpolation=cv2.INTER_CUBIC))

solutions = np.loadtxt('data/training_solutions_rev1.csv', delimiter = ',', skiprows=1)
