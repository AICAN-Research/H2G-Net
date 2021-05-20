import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np


@tf.contrib.eager.function  # Why this decorated is not only just sexy, but smart to use <- https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/function
def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
        y_pred = y_pred[:, 1:2]
        y_true = y_true[:, 1:2]
    return y_true, y_pred


@tf.contrib.eager.function  # <- FIXME: Benchmark this... Is it slower in TF==1.13.1 ?
def precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = true_positives / (possible_positives + K.epsilon())
    return precision


@tf.contrib.eager.function
def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = true_positives / (predicted_positives + K.epsilon())
    return recall


@tf.contrib.eager.function
def f1(y_true, y_pred):
    pr = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((pr * rec) / (pr + rec + K.epsilon()))


# faster implementation in numpy to calculate these respective metrics for large ND-arrays if only one class is of interest, compared to the implementation in scikit-image
# @ TODO: Should handle edge scenarios (when stuff goes to NaN)
def precision_recall_f1score_binary(gt, pred, eps=0):
    true_positives = np.sum(pred * gt)
    possible_positives = np.sum(gt)
    predicted_positives = np.sum(pred)
    pr_ = true_positives / possible_positives
    rec_ = true_positives / predicted_positives
    f1_ = 2 * (pr_ * rec_ / (pr_ + rec_))
    return pr_, rec_, f1_


# Dice score for single class
def dsc(pred, gt, smooth=0):
    intersection1 = np.sum(pred * gt)
    union1 = np.sum(pred * pred) + np.sum(gt * gt)
    return (2. * intersection1 + smooth) / (union1 + smooth)
