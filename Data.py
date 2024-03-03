import os
import numpy as np
#import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data_and_split(path, split=0.1):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "l_mask/*")))

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
#(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data_and_split(path)

def load_data(path):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "l_mask/*")))

    total_size = len(images)
    
    return (images, masks)

#images, masks = load_data(path)