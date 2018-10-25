import cv2
import os
import json
import zipfile
import skimage.io as io
import skimage.transform as trans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def prepare_training_data(training_images_path, training_masks_path, train_test_split_size=0.1):
	os.chdir(training_images_path)
#	train_image_dir = os.chdir(training_images_path)
	train_im = os.listdir('.')
	x1 = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in train_im]) / 255
	x2 = np.flip(x1, 2) # Augmentation
	os.chdir('../..')
	#
	os.chdir(training_masks_path)
#	train_masks_dir = os.chdir(training_masks_path)
	train_ma = os.listdir('.')
	y1 = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in train_ma]) / 255
	y2 = np.flip(y1, 2) # Augmentation
	os.chdir('../..')
	# Augmented data
	x = np.append(x1, x2, axis=0)
	y = np.append(y1, y2, axis=0)
	# expand dimensions for CNN
	x = np.expand_dims(x, axis=3)
	y = np.expand_dims(y, axis=3)
	# Random (train/validation) split 
	x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=train_test_split_size, random_state=46)
	# Non-random (train/validation) split
	train_val_split = 0.05
	x_train = x[0:int(x.shape[0]*(1-train_val_split)),:,:,:]
	y_train = y[0:int(y.shape[0]*(1-train_val_split)),:,:,:]
	x_val = x[int(x.shape[0]*(1-train_val_split)):,:,:,:]
	y_val = y[int(y.shape[0]*(1-train_val_split)):,:,:,:]
	return x_train, y_train, x_val, y_val

def prepare_test_data(test_images_path):
	os.chdir(test_images_path)
#	test_image_dir = os.chdir('/images')
	test_im = os.listdir('.')
	x_test = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in test_im]) / 255
	x_test = np.expand_dims(x_test, axis=3)
	os.chdir('..')
	return x_test


if __name__ == '__main__':
	pass
