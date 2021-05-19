import tensorflow as tf
from tensorflow.python.keras import backend as K


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
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


@tf.contrib.eager.function
def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


@tf.contrib.eager.function
def f1(y_true, y_pred):
    pr = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((pr * rec) / (pr + rec + K.epsilon()))
