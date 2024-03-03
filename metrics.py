import numpy as np
import tensorflow as tf

def iou(y_true, y_pred):
    """
    Calculates the Intersection over Union (IoU) metric for binary segmentation.

    Parameters:
    ----------
    y_true : tf.Tensor
        Ground truth binary mask.
    y_pred : tf.Tensor
        Predicted binary mask.

    Returns:
    -------
    tf.Tensor
        IoU score (intersection over union) for the given masks.
    """
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    iou_local = (intersection + 1e-15) / (union + 1e-15)
    iou_local = iou_local.astype(np.float32)
    return iou_local

def mean_iou(y_true, y_pred):
    """
    Calculates the mean Intersection over Union (IoU) metric for multi-class segmentation.

    Parameters:
    ----------
    y_true : tf.Tensor
        Ground truth segmentation mask (one-hot encoded).
    y_pred : tf.Tensor
        Predicted segmentation mask (one-hot encoded).

    Returns:
    -------
    tf.Tensor
        Mean IoU score across all classes.
    """
    num_classes = y_pred.shape[-1]

    y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes, axis=-1)
    y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=-1), num_classes, axis=-1)

    intersection = tf.math.reduce_sum(y_true * y_pred, axis=(1, 2))
    total = tf.math.reduce_sum(y_true, axis=(1, 2)) + tf.math.reduce_sum(y_pred, axis=(1, 2))
    union = total - intersection

    is_class_present = tf.cast(tf.math.not_equal(total, 0), dtype=tf.float32)
    num_classes_present = tf.math.reduce_sum(is_class_present, axis=1)

    iou_local = tf.math.divide_no_nan(intersection, union)
    iou_local = tf.math.reduce_sum(iou_local, axis=1) / num_classes_present

    mean_iou_local = tf.math.reduce_mean(iou_local)
    return mean_iou_local
