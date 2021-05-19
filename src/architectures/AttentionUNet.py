from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, SpatialDropout2D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape, UpSampling3D, Convolution3D, MaxPooling3D, SpatialDropout3D,\
    Conv2DTranspose, Conv3DTranspose, add, multiply, Reshape, Softmax, AveragePooling3D, Add, Layer
from tensorflow.python.keras.models import Model
import tensorflow as tf
import numpy as np


def get_seed():
    return 420


class PAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        input_shape = x.get_shape().as_list()
        _, h, w, filters = input_shape
        b_layer = Convolution2D(filters // 8, 1, use_bias=False)(x)
        c_layer = Convolution2D(filters // 8, 1, use_bias=False)(x)
        d_layer = Convolution2D(filters, 1, use_bias=False)(x)

        b_layer = tf.transpose(Reshape(target_shape=(h * w, filters // 8))(b_layer), perm=[0, 2, 1])  # @FIXME: Correct for 2D?
        c_layer = Reshape(target_shape=(h * w, filters // 8))(c_layer)
        d_layer = Reshape(target_shape=(h * w, filters))(d_layer)

        # The bc_mul matrix should be of size (H*W*D) * (H*W*D)
        bc_mul = tf.linalg.matmul(c_layer, b_layer)
        activation_bc_mul = Activation(activation='softmax')(bc_mul)
        bcd_mul = tf.linalg.matmul(activation_bc_mul, d_layer)
        bcd_mul = Reshape(target_shape=(h, w, filters))(bcd_mul)
        out = (self.gamma * bcd_mul) + x
        return out


def convolution_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, renorm=False):
    for i in range(2):
        x = Convolution2D(nr_of_convolutions, 3, padding='same')(x)
        if use_bn:
            x = BatchNormalization(renorm=renorm)(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout2D(spatial_dropout, seed=get_seed())(x)

    return x


def attention_block(g, x, nr_of_convolutions, renorm=False):
    """
    Taken from https://github.com/LeeJunHyun/Image_Segmentation
    """
    g1 = Convolution2D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(g)
    g1 = BatchNormalization(renorm=renorm)(g1)

    x1 = Convolution2D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(x)
    x1 = BatchNormalization(renorm=renorm)(x1)

    psi = Concatenate()([g1, x1])
    psi = Activation(activation='relu')(psi)
    psi = Convolution2D(1, kernel_size=1, strides=1, padding='same', use_bias=True)(psi)
    psi = BatchNormalization(renorm=renorm)(psi)
    psi = Activation(activation='sigmoid')(psi)

    return multiply([x, psi])


def attention_block_oktay(g, x, nr_of_convolutions, renorm=False):
    """
    Following Oktay's paper
    """
    g1 = Convolution2D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(g)
    g1 = BatchNormalization(renorm=renorm)(g1)

    x1 = MaxPooling2D([2, 2])(x)
    x1 = Convolution2D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(x1)
    x1 = BatchNormalization(renorm=renorm)(x1)

    psi = Concatenate()([g1, x1])
    psi = Activation(activation='relu')(psi)
    psi = Convolution2D(1, kernel_size=1, strides=1, padding='same', use_bias=True)(psi)
    psi = BatchNormalization(renorm=renorm)(psi)
    psi = Activation(activation='sigmoid')(psi)

    return multiply([x, psi])

def encoder_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, renorm=False):
    x_before_downsampling = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout, renorm=renorm)
    downsample = [2, 2]
    for i in range(1, 3):
        if x.shape[i] <= 3:
            downsample[i-1] = 1

    x = MaxPooling2D(downsample)(x_before_downsampling)

    return x, x_before_downsampling


def encoder_block_pyramid(x, input_ds, nr_of_convolutions, use_bn=False, spatial_dropout=None, renorm=False):
    #pyramid_conv = convolution_block(input_ds, nr_of_convolutions, use_bn, spatial_dropout)
    pyramid_conv = Convolution2D(filters=nr_of_convolutions, kernel_size=(3, 3), padding='same', activation='relu')(input_ds)
    x = Concatenate(axis=-1)([pyramid_conv, x])
    x_before_downsampling = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout, renorm=renorm)
    downsample = [2, 2]
    for i in range(1, 3):
        if x.shape[i] <= 4:
            downsample[i-1] = 1

    x = MaxPooling2D(downsample)(x_before_downsampling)

    return x, x_before_downsampling


def decoder_block(x, cross_over_connection, nr_of_convolutions, use_bn=False, spatial_dropout=None, renorm=False):
    x = Conv2DTranspose(nr_of_convolutions, kernel_size=3, padding='same', strides=2)(x)
    if use_bn:
        x = BatchNormalization(renorm=renorm)(x)
    x = Activation('relu')(x)
    attention = attention_block(g=x, x=cross_over_connection, nr_of_convolutions=int(nr_of_convolutions/2), renorm=renorm)
    # pam = PAM()(attention)
    # x = Concatenate()([x, attention, pam])
    x = Concatenate()([x, attention])
    x = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout, renorm=renorm)

    return x

def decoder_block_oktay(x, cross_over_connection, nr_of_convolutions, use_bn=False, spatial_dropout=None, renorm=False):
    x_down = x
    x = Conv2DTranspose(nr_of_convolutions, kernel_size=3, padding='same', strides=2)(x)
    if use_bn:
        x = BatchNormalization(renorm=renorm)(x)
    x = Activation('relu')(x)
    attention = attention_block_oktay(g=x, x=cross_over_connection, nr_of_convolutions=int(nr_of_convolutions/2), renorm=renorm)
    x = Concatenate()([x, attention])
    x = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout, renorm=renorm)

    return x

class AttentionUnet():
    def __init__(self, input_shape, nb_classes, deep_supervision=False, input_pyramid=False):
        if len(input_shape) != 3 and len(input_shape) != 4:
            raise ValueError('Input shape must have 3 or 4 dimensions')
        if nb_classes <= 1:
            raise ValueError('Segmentation classes must be > 1')
        self.dims = 2
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.deep_supervision = deep_supervision
        self.input_pyramid = input_pyramid
        self.convolutions = None
        self.encoder_use_bn = True
        self.decoder_use_bn = True
        self.encoder_spatial_dropout = None
        self.decoder_spatial_dropout = None
        self.renorm = False

    def set_renorm(self, value):
        self.renorm = value

    def set_convolutions(self, convolutions):
        self.convolutions = convolutions

    def get_dice_loss(self):
        def dice_loss(target, output, epsilon=1e-10):
            smooth = 1.
            dice = 0

            #for object in range(1, self.nb_classes):
            for object in range(0, self.nb_classes):
                if self.dims == 2:
                    output1 = output[:, :, :, object]
                    target1 = target[:, :, :, object]
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
                intersection1 = tf.reduce_sum(output1 * target1)
                union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
                dice += (2. * intersection1 + smooth) / (union1 + smooth)

            dice /= (self.nb_classes - 1)

            return tf.clip_by_value(1. - dice, 0., 1. - epsilon)

        return dice_loss

    def get_dice_metric_no_bg(self):
        def get_dice_metric_no_bg(target, output, epsilon=1e-10):
            smooth = 1.
            dice = 0

            for object in range(1, self.nb_classes):
                if self.dims == 2:
                    output1 = output[:, :, :, object]
                    target1 = target[:, :, :, object]
                    shapes = (1, 2)
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
                    shapes = (1, 2, 3)
                intersection1 = tf.reduce_sum(output1 * target1, axis=shapes)
                union1 = tf.reduce_sum(output1 * output1, axis=shapes) + tf.reduce_sum(target1 * target1, axis=shapes)
                dice += (2. * intersection1 + smooth) / (union1 + smooth)

            dice /= (self.nb_classes - 1)

            return tf.clip_by_value(dice, 0., 1. - epsilon)

        return get_dice_metric_no_bg

    def get_dice_loss_no_bg(self):
        def get_dice_loss_no_bg(target, output, epsilon=1e-10):
            smooth = 1.
            dice = 0

            for object in range(1, self.nb_classes):
                if self.dims == 2:
                    output1 = output[:, :, :, object]
                    target1 = target[:, :, :, object]
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
                intersection1 = tf.reduce_sum(output1 * target1)
                union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
                dice += (2. * intersection1 + smooth) / (union1 + smooth)

            dice /= (self.nb_classes - 1)

            return tf.clip_by_value(1. - dice, 0., 1. - epsilon)

        return get_dice_loss_no_bg

    def get_dice_metric(self, use_background=False):
        def dice(target, output, epsilon=1e-10):
            smooth = 0  # 1e-8  # 1.
            dice = 0

            # first threshold output volume
            output = tf.cast(tf.math.greater_equal(output, tf.constant([0.5])), tf.float32)

            for object in range(0 if use_background else 1, self.nb_classes):
                if self.dims == 2:
                    output1 = output[:, :, :, object]
                    target1 = target[:, :, :, object]
                    shapes = (1, 2)
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
                    shapes = (1, 2, 3)
                intersection1 = tf.reduce_sum(output1 * target1, axis=shapes)
                union1 = tf.reduce_sum(output1 * output1, axis=shapes) + tf.reduce_sum(target1 * target1, axis=shapes)
                dice += (2. * intersection1 + smooth) / (union1 + smooth)

            if use_background:
                dice /= self.nb_classes
            else:
                dice /= (self.nb_classes - 1)

            return dice  # tf.clip_by_value(dice, 0., 1. - epsilon)

        return dice

    def get_focal_dice_loss_no_bg(self):
        def get_focal_dice_loss_no_bg(target, output, epsilon=1e-10, smooth=1):
            smooth = 1.
            dice = 0

            for object in range(1, self.nb_classes):
                if self.dims == 2:
                    output1 = output[:, :, :, object]
                    target1 = target[:, :, :, object]
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
                intersection1 = tf.reduce_sum(output1 * target1)
                union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
                dice += (2. * intersection1 + smooth) / (union1 + smooth)

            dice /= (self.nb_classes - 1)
            gamma = 2.
            focal_dice = tf.math.pow((1. - dice), gamma)
            #return tf.clip_by_value(1. - tversky, 0., 1. - epsilon)
            return focal_dice
        return get_focal_dice_loss_no_bg

    def get_tversky_loss_no_bg(self):
        def get_tversky_loss_no_bg(target, output, epsilon=1e-10, smooth=1):
            for object in range(1, self.nb_classes):
                if self.dims == 2:
                    output1 = output[:, :, :, object]
                    target1 = target[:, :, :, object]
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
            y_true_pos = target1
            y_pred_pos = output1
            true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
            false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
            false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
            alpha = 0.7
            tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

            #return tf.clip_by_value(1. - tversky, 0., 1. - epsilon)
            return 1. - tversky
        return get_tversky_loss_no_bg

    def get_focal_tversky_loss_no_bg(self):
        def get_focal_tversky_loss_no_bg(target, output, epsilon=1e-10, smooth=1):
            for object in range(1, self.nb_classes):
                if self.dims == 2:
                    output1 = output[:, :, :, object]
                    target1 = target[:, :, :, object]
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
            y_true_pos = target1
            y_pred_pos = output1
            true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
            false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
            false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
            alpha = 0.7
            tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + (1. - alpha) * false_pos + smooth)
            gamma = 2.
            #focal_tversky = tf.python.keras.backend.pow((1. - tversky), gamma)
            focal_tversky = tf.math.pow((1. - tversky), gamma)
            #return tf.clip_by_value(1. - tversky, 0., 1. - epsilon)
            return focal_tversky
        return get_focal_tversky_loss_no_bg

    def create(self):
        """
        Create model and return it

        :return: keras model
        """

        input_layer = Input(shape=self.input_shape)
        x = input_layer

        init_size = max(self.input_shape[:-1])
        size = init_size

        convolutions = self.convolutions
        connection = []
        i = 0

        if self.input_pyramid:
            scaled_input = []
            scaled_input.append(x)
            for i, nbc in enumerate(self.convolutions[:-1]):
                ds_input = AveragePooling2D(pool_size=(2, 2))(scaled_input[i])
                scaled_input.append(ds_input)

        for i, nbc in enumerate(self.convolutions[:-1]):
            if not self.input_pyramid or (i == 0):
                x, x_before_ds = encoder_block(x, nbc, use_bn=self.encoder_use_bn, spatial_dropout=self.encoder_spatial_dropout, renorm=self.renorm)
            else:
                x, x_before_ds = encoder_block_pyramid(x, scaled_input[i], nbc, use_bn=self.encoder_use_bn, spatial_dropout=self.encoder_spatial_dropout, renorm=self.renorm)
            connection.insert(0, x_before_ds)  # Append in reverse order for easier use in the next block

        x = convolution_block(x, self.convolutions[-1], self.encoder_use_bn, self.encoder_spatial_dropout, renorm=self.renorm)
        connection.insert(0, x)

        inverse_conv = self.convolutions[::-1]
        inverse_conv = inverse_conv[1:]
        decoded_layers = []
        #@TODO. Should Attention Gating be done over the last feature map (i.e. image at the highest resolution)?
        # Some papers say they don't because the feature map does not represent the data in a high dimensional space.
        for i, nbc in enumerate(inverse_conv):
            x = decoder_block(x, connection[i+1], nbc, use_bn=self.decoder_use_bn, spatial_dropout=self.decoder_spatial_dropout, renorm=self.renorm)
            decoded_layers.append(x)

        if not self.deep_supervision:
            # Final activation layer
            x = Convolution2D(self.nb_classes, 1, activation='softmax')(x)
        else:
            recons_list = []
            for i, lay in enumerate(decoded_layers):
                x = Convolution2D(self.nb_classes, 1, activation='softmax')(lay)
                recons_list.append(x)
            x = recons_list[::-1]

        return Model(inputs=input_layer, outputs=x)


if __name__ == "__main__":
    network = AttentionUnet(input_shape=(1024, 1024, 4), nb_classes=2, deep_supervision=True, input_pyramid=True)
    network.decoder_dropout = 0.1
    network.set_renorm(True)
    network.set_convolutions([8, 16, 32, 64, 128, 128, 256, 256])
    model = network.create()
