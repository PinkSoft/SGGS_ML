import tensorflow as tf
import numpy as np
import cv2

# Dictionary mapping class IDs to colors.
id2color = {
    0: (0, 0, 0),  # Background
    1: (255, 0, 0),  # lumen
    2: (0, 0, 255),  # stenose
}

def read_image_mask(image_path, mask=False, size=(512, 512)):
    """
    Read and preprocess an image or mask.

    Args:
        image_path (str): Path to the image file.
        mask (bool, optional): Whether the input is a mask. Defaults to False.
        size (tuple, optional): Desired image size (height, width). Defaults to (512, 512).

    Returns:
        tf.Tensor: Processed image or mask.
    """
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.io.decode_image(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=size, method="bicubic")

        image_mask = tf.zeros_like(image)
        cond = image >= 200
        updates = tf.ones_like(image[cond])
        image_mask = tf.tensor_scatter_nd_update(image_mask, tf.where(cond), updates)
        image = tf.cast(image_mask, tf.uint8)

    else:
        image = tf.io.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=size, method="bicubic")
        image = tf.cast(tf.clip_by_value(image, 0.0, 255.0), tf.float32)

    return image

def load_data(image_list, mask_list, mask=False, size=(512, 512)):
    """
    Load image and mask data.

    Args:
        image_list (list): List of image file paths.
        mask_list (list): List of mask file paths.
        mask (bool, optional): Whether to load masks. Defaults to False.
        size (tuple, optional): Desired image size (height, width). Defaults to (512, 512).

    Returns:
        dict: A dictionary containing 'images' and 'segmentation_masks'.
    """
    image = read_image_mask(image_list, size=size)
    mask = read_image_mask(mask_list, mask=True, size=size)
    return {"images": image, "segmentation_masks": mask}

def unpackage_inputs(inputs):
    """
    Unpackage inputs from a dictionary.

    Args:
        inputs (dict): A dictionary containing 'images' and 'segmentation_masks'.

    Returns:
        tuple: Tuple of images and segmentation masks.
    """
    images = inputs["images"]
    segmentation_masks = inputs["segmentation_masks"]
    return images, segmentation_masks

def num_to_rgb(num_arr, color_map=id2color):
    """
    Convert a single-channel mask representation to an RGB mask.

    Args:
        num_arr (np.ndarray): Single-channel mask array.
        color_map (dict, optional): Mapping of class IDs to RGB colors. Defaults to id2color.

    Returns:
        np.ndarray: RGB mask.
    """
    output = np.zeros(num_arr.shape[:2] + (3,))

    for k in color_map.keys():
        output[num_arr == k] = color_map[k]

    return output.astype(np.uint8)

def image_overlay(image, segmented_image):
    """
    Overlay a segmentation map on top of an RGB image.

    Args:
        image (np.ndarray): Original RGB image.
        segmented_image (np.ndarray): Segmentation map (single-channel or RGB).

    Returns:
        np.ndarray: Overlay image.
    """
    alpha = 1.0  # Transparency for the original image.
    beta = 0.7  # Transparency for the segmentation map.
    gamma = 0.0  # Scalar added to each sum.

    image = image.astype(np.uint8)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
