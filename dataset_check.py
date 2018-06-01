import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from own_data_preprocessing import own_data_preprocessing
import tensorflow as tf
import os

#train_set_x_orig, train_set_y, test_set_x_orig, test_set_y = own_data_preprocessing()

path = 'C:\\gugu\\project\\Own data preprocessing\\resize'
img_list = os.listdir(path)

number_of_samples = len(img_list)

for file in img_list:
    print(file)
