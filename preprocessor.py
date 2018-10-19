import cv2
import os
import json
import zipfile
import skimage.io as io
import skimage.transform as trans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##### Use kaggle API
!pip install kaggle
api_token = {"username":"des137","key":"##########"}
os.chdir('/')
!mkdir ~/.kaggle #kaggle API searches in root directory for .kaggle/kaggle.json
with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

!chmod 600 /root/.kaggle/kaggle.json

# API link from Kaggle:

!kaggle competitions download -c tgs-salt-identification-challenge
zip_ref = zipfile.ZipFile('train.zip', 'r')
zip_ref.extractall()
zip_ref.close()

##### Prepare data
tr_image_dir = os.chdir('/images')
train_im = os.listdir(tr_image_dir)
x1 = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in train_im]) / 255
x2 = np.flip(x1, 2)
tr_masks_dir = os.chdir('/masks')
train_ma = os.listdir(tr_masks_dir)
y1 = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in train_ma]) / 255
y2 = np.flip(y1, 2)
# expand dimensions for CNN

x = np.append(x1, x2, axis=0)
y = np.append(y1, y2, axis=0)

x = np.expand_dims(x, axis=3)
y = np.expand_dims(y, axis=3)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.025, random_state=46)

# split training vs validation set
train_val_split = 0.05
x_train = x[0:int(x.shape[0]*(1-train_val_split)),:,:,:]
y_train = y[0:int(y.shape[0]*(1-train_val_split)),:,:,:]
x_val = x[int(x.shape[0]*(1-train_val_split)):,:,:,:]
y_val = y[int(y.shape[0]*(1-train_val_split)):,:,:,:]


os.chdir('/')
!mkdir train
!mv images train/images

zip_ref = zipfile.ZipFile('test.zip', 'r')
zip_ref.extractall()
zip_ref.close()

test_image_dir = os.chdir('/images')
test_im = os.listdir(test_image_dir)
x_test = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in test_im]) / 255
x_test = np.expand_dims(x_test, axis=3)
x_test_pred = model.predict(x_test, verbose=1)
x_test_final = np.round(x_test_pred[:,:,:,0])
