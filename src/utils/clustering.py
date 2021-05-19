import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec


class StandardizerLayer(Layer):
    def __init__(self, weights=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(StandardizerLayer, self).__init__(**kwargs)
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        input_dim = input_shape[1].value
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.vars = self.add_weight(shape=(input_dim, 2), initializer='ones', name='statistics_', trainable=False)
        self.built = True

    def call(self, features, **kwargs):
        return (features - self.vars[:, 0]) / self.vars[:, 1]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]


class PCALayer(Layer):
    def __init__(self, weights=None, pca_features=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(PCALayer, self).__init__(**kwargs)
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
        self.num_pca_features = pca_features  # 3331  # InceptionV3: 3331, MobileNetV2: 2461

    def build(self, input_shape):
        input_dim = input_shape[1].value
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.pca_matrix = self.add_weight(shape=(input_dim, self.num_pca_features), initializer='ones', name='pca_matrix_', trainable=False)
        self.pca_mean_ = self.add_weight(shape=(1, input_dim), initializer='zeros', name='pca_mean_', trainable=False)
        self.built = True

    def call(self, features, **kwargs):
        features -= self.pca_mean_
        return K.dot(features, self.pca_matrix)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.num_pca_features


class KmeansClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(KmeansClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
        self.centroids = None

    def build(self, input_shape):
        input_dim = input_shape[1].value
        self.input_dim = input_dim
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.centroids = self.add_weight(shape=(input_dim, self.n_clusters), initializer='zeros', name='clusters_', trainable=False)
        self.built = True

    def call(self, features, **kwargs):
        distances = tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(features, -1), tf.expand_dims(self.centroids, 0))), 1)
        assignments = tf.argmin(distances, -1)
        assignments = tf.expand_dims(assignments, -1)
        assignments = tf.cast(assignments, tf.float32)
        return assignments

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(KmeansClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))