import os
#import cv2
from glob import glob
from sklearn.model_selection import train_test_split

def load_data_and_split(path, mask_set = 'Multi', split=0.1):
    """
    Load image and mask data from the specified path and split it into 
    train, validation, and test sets.

    Args:
        path (str): Path to the directory containing image and mask files.
        mask_set (str): Type of mask which has to be added tor the data 
                                Default to 'Multi'
        split (float, optional): Proportion of data to allocate for 
                                validation and test sets. Defaults to 0.1.

    Returns:
        tuple: Tuple containing train, validation, and test data:
            - train_x (list): List of file paths for training images.
            - train_y (list): List of file paths for corresponding training masks.
            - valid_x (list): List of file paths for validation images.
            - valid_y (list): List of file paths for corresponding validation masks.
            - test_x (list): List of file paths for test images.
            - test_y (list): List of file paths for corresponding test masks.
    """
    images = sorted(glob(os.path.join(path, "images/*")))
    if mask_set == 'Lumen':
        masks = sorted(glob(os.path.join(path, "l_masks/*")))
    elif mask_set == 'Stenosis':
        masks = sorted(glob(os.path.join(path, "s_masks/*")))
    else:
        masks = sorted(glob(os.path.join(path, "masks/*")))

    total_size = len(images)
    split_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def load_data(path, mask_set = 'Multi'):
    """
    Load image and mask data from the specified path.

    Args:
        path (str): Path to the directory containing image and mask files.
        mask_set (str): Type of mask which has to be added tor the data 
                                Default to 'Multi'

    Returns:
        tuple: Tuple containing image and mask data:
            - images (list): List of file paths for images.
            - masks (list): List of file paths for corresponding masks.
    """
    images = sorted(glob(os.path.join(path, "images/*")))
    if mask_set == 'Lumen':
        masks = sorted(glob(os.path.join(path, "l_masks/*")))
    elif mask_set == 'Stenosis':
        masks = sorted(glob(os.path.join(path, "s_masks/*")))
    else:
        masks = sorted(glob(os.path.join(path, "masks/*")))
    return (images, masks)
