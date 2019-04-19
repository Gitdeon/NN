import glob
import os
import numpy as np
import matplotlib as plt
import cv2 #pip3 install opencv-python
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID = 0

jpg_paths = glob.glob('data/images_training_rev1/*.jpg')
jpg_paths = np.sort(jpg_paths)

#Loading all images
galaxy_images = []
for i in range(len(jpg_paths)):
    jpg = cv2.imread(jpg_paths[i])
    galaxy_images.append(cv2.resize(jpg, dsize=(128, 128), interpolation=cv2.INTER_CUBIC))
    if i % 1000 == 0: print('Loaded ', i, 'images.')

#Get predicitions
solutions = np.loadtxt('data/training_solutions_rev1.csv', delimiter = ',', skiprows=1)
classification_solutions = []
for i in range(len(solutions)):
    classification_solutions.append([solutions[i][0],np.argmax(solutions[i][1:])])

images_train, images_test, solutions_train, solutions_test = train_test_split(galaxy_images, classification_solutions, test_size=0.2, random_state=42)
