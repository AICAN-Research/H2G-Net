import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import *


def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


def conv_block(inputs, filters):
    x = inputs

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization(renorm=True)(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.1)(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization(renorm=True)(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.1)(x)

    x = squeeze_excite_block(x)

    return x


def encoder1(inputs, weights="imagenet"):
    skip_connections = []

    model = VGG19(include_top=False, weights=weights, input_tensor=inputs)
    names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
    for name in names:
        skip_connections.append(model.get_layer(name).output)

    output = model.get_layer("block5_conv4").output
    return output, skip_connections


def decoder1(inputs, skip_connections, num_filters=[256, 128, 64, 32]):
    #num_filters = [256, 128, 64, 32]
    skip_connections.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_connections[i]])
        x = conv_block(x, f)

    return x


# def encoder2(inputs):
#     skip_connections = []
#
#     output = DenseNet121(include_top=False, weights='imagenet')(inputs)
#     model = tf.keras.models.Model(inputs, output)
#
#     names = ["input_2", "conv1/relu", "pool2_conv", "pool3_conv"]
#     for name in names:
#         skip_connections.append(model.get_layer(name).output)
#     output = model.get_layer("pool4_conv").output
#
#     return output, skip_connections


def encoder2(inputs, num_filters=[32, 64, 128, 256]):
    # num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)

    return x, skip_connections


def decoder2(inputs, skip_1, skip_2, num_filters=[256, 128, 64, 32]):
    #num_filters = [256, 128, 64, 32]
    skip_2.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_1[i], skip_2[i]])
        x = conv_block(x, f)

    return x


def output_block(inputs, nb_classes, name):
    x = Conv2D(nb_classes, (1, 1), padding="same")(inputs)
    x = Activation('softmax', name=name)(x)
    return x


def Upsample(tensor, size):
    """Bilinear upsampling"""
    def _upsample(x, size):
        return tf.image.resize(images=x, size=size)
    return Lambda(lambda x: _upsample(x, size), output_shape=size)(tensor)


def ASPP(x, filter):
    shape = x.shape

    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization(renorm=True)(y1)
    y1 = Activation("relu")(y1)
    y1 = SpatialDropout2D(0.1)(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)

    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization(renorm=True)(y2)
    y2 = Activation("relu")(y2)
    y2 = SpatialDropout2D(0.1)(y2)

    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization(renorm=True)(y3)
    y3 = Activation("relu")(y3)
    y3 = SpatialDropout2D(0.1)(y3)

    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization(renorm=True)(y4)
    y4 = Activation("relu")(y4)
    y4 = SpatialDropout2D(0.1)(y4)

    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization(renorm=True)(y5)
    y5 = Activation("relu")(y5)
    y5 = SpatialDropout2D(0.1)(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization(renorm=True)(y)
    y = Activation("relu")(y)
    y = SpatialDropout2D(0.1)(y)

    return y


class ExtractChannel(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExtractChannel, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.expand_dims(inputs[..., -1], axis=-1)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(ExtractChannel, self).get_config()
        config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))


def build_model(input_shape, nb_classes, weights="imagenet", encoder_filters=[32, 64, 128, 256], decoder_filters=[256, 128, 64, 32], aspp_filter=64, outname1="unet", outname2="refinement"):
    inputs = Input(input_shape)
    #x, skip_1 = encoder1(inputs, weights, encoder_filters)  # VGG19 encoder
    x, skip_1 = encoder2(inputs, encoder_filters)
    x = ASPP(x, aspp_filter)
    x = decoder1(x, skip_1, decoder_filters)
    outputs1 = output_block(x, nb_classes, name=outname1)

    # FIXME: Andre had to introduce small fix here, but this only works for binary classification currently!
    #print(inputs.shape, outputs1.shape)

    #tmp = outputs1
    #tmp = ExtractChannel()(outputs1)
    #x = inputs * tmp  # only filter based on class of interest
    x = tf.keras.layers.Concatenate()([inputs, outputs1])  # for multi-class case, have to do this

    x, skip_2 = encoder2(x, encoder_filters)
    x = ASPP(x, aspp_filter)
    x = decoder2(x, skip_1, skip_2, decoder_filters)
    outputs2 = output_block(x, nb_classes, name=outname2)
    #outputs = tf.keras.layers.Concatenate()([outputs1, outputs2])

    model = Model(inputs=inputs, outputs=[outputs1, outputs2])

    return model


if __name__ == "__main__":
    encoder_filters = [8, 16, 32, 64, 128, 128, 256, 256]
    aspp_filter = 32
    model = build_model(input_shape=(1024, 1024, 4), nb_classes=2, weights=None, encoder_filters=encoder_filters,
                        decoder_filters=encoder_filters[::-1], aspp_filter=aspp_filter)
