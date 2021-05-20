import tensorflow as tf
from tensorflow.python.keras import backend as K


# @tf.contrib.eager.function
def cluster_weighted_loss_metric(ys_true, ys_pred, nb_classes=2, nb_clusters=10):
    ys_true = [ys_true[:, :nb_classes], ys_true[:, nb_classes:]]
    ys_pred = [ys_pred[:, :nb_classes], ys_pred[:, nb_classes:]]

    filter_cluster_batch = tf.squeeze(ys_pred[1], axis=1)  # better for sigmoid predictions?

    # for both softmax classifications and clusterings
    ys_pred[0] /= K.sum(ys_pred[0], axis=-1, keepdims=True)  # scale predictions so that the class probas of each sample sum to 1
    ys_pred[0] = K.clip(ys_pred[0], K.epsilon(), 1 - K.epsilon())  # clip to prevent NaN's and Inf's

    # calculate loss for each class (to do macro average)
    loss = 0
    for c in range(nb_classes):
        # filter all relevant tensors to only work with tensors belonging to current class
        filter_class = tf.equal(ys_true[0][:, c], 1)
        ys_pred_class_curr_class = tf.boolean_mask(ys_pred[0], filter_class, axis=0)
        ys_true_class_curr_class = tf.boolean_mask(ys_true[0], filter_class, axis=0)
        filter_batch_class_filtered = tf.boolean_mask(filter_cluster_batch, filter_class)

        # need to calculate weights for current set of samples (in batch) for each class
        ys_pred_curr_onehot = tf.one_hot(tf.cast(filter_batch_class_filtered, tf.int32), depth=nb_clusters)
        counts = tf.reduce_sum(ys_pred_curr_onehot, axis=0)

        # substitute 0-counts with 1, to avoid dividing by zero
        counts_orig = tf.identity(counts)
        counts = tf.where(tf.equal(counts, 0), tf.ones_like(counts), counts)

        # calculate loss for each cluster (to do macro average)
        cluster_loss = 0
        for i in range(nb_clusters):
            # only keep samples from the current cluster in the current class
            filter_cluster_in_class = tf.equal(filter_batch_class_filtered, i)
            val1 = tf.boolean_mask(ys_true_class_curr_class, filter_cluster_in_class, axis=0)
            val2 = tf.boolean_mask(ys_pred_class_curr_class, filter_cluster_in_class, axis=0)

            val1 = K.clip(val1, K.epsilon(), 1 - K.epsilon())
            val2 = K.clip(val2, K.epsilon(), 1 - K.epsilon())
            cluster_loss += tf.reduce_sum(val1 * K.log(val2)) / counts[i]

        cluster_loss /= tf.cast(tf.count_nonzero(counts_orig), tf.float32)
        loss += cluster_loss
    loss /= nb_classes
    return -loss
