from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, SpatialDropout2D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape, UpSampling3D, Convolution3D, MaxPooling3D, SpatialDropout3D,\
    Conv2DTranspose, Conv3DTranspose, add, multiply, Reshape, Softmax, AveragePooling3D, Add, Layer
from tensorflow.python.keras.models import Model
import tensorflow as tf
import numpy as np
import math
#from functions import get_seed


def get_seed():
    return 420


class CAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(CAM, self).__init__(**kwargs)
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

        vec_a = Reshape(target_shape=(h * w, filters))(x)
        vec_aT = tf.transpose(vec_a, perm=[0, 2, 1])
        aTa = tf.linalg.matmul(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)
        aaTa = tf.linalg.matmul(vec_a, softmax_aTa)
        aaTa = Reshape(target_shape=(h, w, filters))(aaTa)
        out = (self.gamma * aaTa) + x
        return out


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

        b_layer = tf.transpose(Reshape(target_shape=(h * w, filters // 8))(b_layer), perm=[0, 2, 1])  # @TODO: it the perm correct here?
        c_layer = Reshape(target_shape=(h * w, filters // 8))(c_layer)
        d_layer = Reshape(target_shape=(h * w, filters))(d_layer)

        # The bc_mul matrix should be of size (H*W*D) * (H*W*D)
        bc_mul = tf.linalg.matmul(c_layer, b_layer)
        activation_bc_mul = Activation(activation='softmax')(bc_mul)
        bcd_mul = tf.linalg.matmul(activation_bc_mul, d_layer)
        bcd_mul = Reshape(target_shape=(h, w, filters))(bcd_mul)
        out = (self.gamma * bcd_mul) + x
        return out


def convolution_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    for i in range(2):
        x = Convolution2D(nr_of_convolutions, 3, padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout2D(spatial_dropout, seed=get_seed())(x)

    return x


def attention_block(g, x, nr_of_convolutions):
    """
    Taken from https://github.com/LeeJunHyun/Image_Segmentation
    """
    g1 = Convolution2D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(g)
    g1 = BatchNormalization()(g1)

    x1 = Convolution2D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(x)
    x1 = BatchNormalization()(x1)

    psi = Concatenate()([g1, x1])
    psi = Activation(activation='relu')(psi)
    #psi = Convolution2D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(psi)
    psi = Convolution2D(1, kernel_size=1, strides=1, padding='same', use_bias=True)(psi)
    psi = BatchNormalization()(psi)
    psi = Activation(activation='sigmoid')(psi)

    return multiply([x, psi])


def channel_attention_block(x):
    # """
    # Taken from https://github.com/junfu1115/DANet/blob/master/encoding/nn/da_att.py
    # """
    # input_shape = tf.shape(x)
    # x_channel_first = tf.transpose(x, perm=[0, 4, 1, 2, 3])
    # proj_query = Reshape(target_shape=(input_shape[0], input_shape[4], input_shape[1] * input_shape[2] * input_shape[3]))(x_channel_first)
    # proj_key = Reshape(target_shape=(input_shape[0], input_shape[4], input_shape[1] * input_shape[2] * input_shape[3]))(x_channel_first)
    # proj_key = tf.transpose(proj_key, perm=[0, 2, 1])
    # # Torch: energy = torch.bmm(proj_query, proj_key)
    # energy = tf.linalg.matmul(proj_query, proj_key)  # or batch_dot?
    # # Torch: energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
    # energy_new = tf.math.reduce_max(energy, axis=-1, keep_dims=True)
    # energy_new = tf.broadcast_to(energy_new, shape=energy.shape)
    # energy_new = energy_new - energy
    # attention = Softmax()(energy_new)
    # proj_value = Reshape(target_shape=(input_shape[0], input_shape[4], -1))(x_channel_first)
    # out = tf.linalg.matmul(attention, proj_value)
    # out = Reshape(target_shape=(input_shape[0], input_shape[4], input_shape[1], input_shape[2], input_shape[3]))(out)
    # gamma = tf.Variable()
    # out = (gamma * out) + x
    # out_channel_last = tf.transpose(out, perm=[0, 2, 3, 4, 1])
    #
    # return out_channel_last

    """
    Taken from https://github.com/niecongchong/DANet-keras/blob/master/layers/attention.py
    """
    input_shape = x.get_shape().as_list()
    _, h, w, filters = input_shape

    vec_a = Reshape(target_shape=(h * w, filters))(x)
    vec_aT = tf.transpose(vec_a, perm=[0, 2, 1])
    aTa = tf.linalg.matmul(vec_aT, vec_a)
    softmax_aTa = Activation('softmax')(aTa)
    aaTa = tf.linalg.matmul(vec_a, softmax_aTa)
    aaTa = Reshape(target_shape=(h, w, filters))(aaTa)
    gamma = tf.Variable(initial_value=1.)
    out = (gamma * aaTa) + x
    return out

def position_attention_module(x, nb_convolutions):
    """
    Taken from https://github.com/junfu1115/DANet/blob/master/encoding/nn/da_att.py
    """
    input_shape = x.get_shape().as_list()
    _, h, w, filters = input_shape
    b_layer = Convolution2D(filters // 8, 1, use_bias=False)(x)
    c_layer = Convolution2D(filters // 8, 1, use_bias=False)(x)
    d_layer = Convolution2D(filters, 1, use_bias=False)(x)

    b_layer = tf.transpose(Reshape(target_shape=(h * w, filters // 8))(b_layer), perm=[0, 2, 1])
    c_layer = Reshape(target_shape=(h * w , filters // 8))(c_layer)
    d_layer = Reshape(target_shape=(h * w, filters))(d_layer)

    # The bc_mul matrix should be of size (H*W*D) * (H*W*D)
    bc_mul = tf.linalg.matmul(c_layer, b_layer)
    activation_bc_mul = Activation(activation='softmax')(bc_mul)
    bcd_mul = tf.linalg.matmul(activation_bc_mul, d_layer)
    bcd_mul = Reshape(target_shape=(h, w, filters))(bcd_mul)

    gamma = tf.Variable(initial_value=1.)
    out = (gamma * bcd_mul) + x
    return out


def encoder_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    x_before_downsampling = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout)
    downsample = [2, 2]
    for i in range(1, 3):
        if x.shape[i] <= 4:
            downsample[i-1] = 1

    x = MaxPooling2D(downsample)(x_before_downsampling)

    return x, x_before_downsampling


def encoder_block_pyramid(x, input_ds, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    #pyramid_conv = convolution_block(input_ds, nr_of_convolutions, use_bn, spatial_dropout)
    pyramid_conv = Convolution2D(filters=nr_of_convolutions, kernel_size=(3, 3), padding='same', activation='relu')(input_ds)
    x = Concatenate(axis=-1)([pyramid_conv, x])
    x_before_downsampling = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout)
    downsample = [2, 2]
    for i in range(1, 3):
        if x.shape[i] <= 4:
            downsample[i-1] = 1

    x = MaxPooling2D(downsample)(x_before_downsampling)

    return x, x_before_downsampling


def decoder_block(x, cross_over_connection, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    x = Conv2DTranspose(nr_of_convolutions, kernel_size=3, padding='same', strides=2)(x)
    x = Concatenate()([cross_over_connection, x])
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout)

    return x


# @TODO. When having the attention_layer, should not concatenate at the first decoder step because x and attention_layer
# are actually the same. Should check the iteration number and only do if not the bottom iteration value?
def decoder_block_guided(x, cross_over_connection, nr_of_convolutions, iteration, attention_layer, use_bn=False, spatial_dropout=None):
    x = Conv2DTranspose(nr_of_convolutions, kernel_size=3, padding='same', strides=2)(x)
    upsampling_factor = int(math.pow(2, iteration))
    attention_layer_up = Conv2DTranspose(nr_of_convolutions, kernel_size=3, padding='same', strides=upsampling_factor)(attention_layer)
    x = Concatenate()([attention_layer_up, cross_over_connection, x])
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout)

    return x


class DualAttentionUnet():
    def __init__(self, input_shape, nb_classes, deep_supervision=False, input_pyramid=False, attention_guiding=False):
        if len(input_shape) != 3 and len(input_shape) != 4:
            raise ValueError('Input shape must have 3 or 4 dimensions')
        if nb_classes <= 1:
            raise ValueError('Segmentation classes must be > 1')
        self.dims = 2
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.deep_supervision = deep_supervision
        self.input_pyramid = input_pyramid
        self.attention_guided = attention_guiding
        self.convolutions = None
        self.encoder_use_bn = True
        self.decoder_use_bn = True
        self.encoder_spatial_dropout = None
        self.decoder_spatial_dropout = None

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
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
                intersection1 = tf.reduce_sum(output1 * target1)
                union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
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
                    shapes = (1, 2)
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
                    shapes = (1, 2, 3)

                intersection1 = tf.reduce_sum(output1 * target1, axis=shapes)
                union1 = tf.reduce_sum(output1 * output1, axis=shapes) + tf.reduce_sum(target1 * target1, axis=shapes)
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
                x, x_before_ds = encoder_block(x, nbc, use_bn=self.encoder_use_bn, spatial_dropout=self.encoder_spatial_dropout)
            else:
                x, x_before_ds = encoder_block_pyramid(x, scaled_input[i], nbc, use_bn=self.encoder_use_bn, spatial_dropout=self.encoder_spatial_dropout)
            connection.insert(0, x_before_ds)  # Append in reverse order for easier use in the next block

        x = convolution_block(x, self.convolutions[-1], self.encoder_use_bn, self.encoder_spatial_dropout)
        connection.insert(0, x)

        # Dual attention scheme, the second conv3D at the end of each module is not in the original paper
        # And they do regular dropout with 0.1.
        # The last SpatialDropout is also not described in the paper, should we drop it?
        pam = PAM()(x)
        pam = Convolution2D(self.convolutions[-1], 3, padding='same')(pam)
        pam = BatchNormalization()(pam)
        pam = Activation('relu')(pam)
        pam = SpatialDropout2D(0.5, seed=get_seed())(pam)
        pam = Convolution2D(self.convolutions[-1], 3, padding='same')(pam)

        cam = CAM()(x)
        cam = Convolution2D(self.convolutions[-1], 3, padding='same')(cam)
        cam = BatchNormalization()(cam)
        cam = Activation('relu')(cam)
        cam = SpatialDropout2D(0.5, seed=get_seed())(cam)
        cam = Convolution2D(self.convolutions[-1], 3, padding='same')(cam)

        x = add([pam, cam])
        x = SpatialDropout2D(0.5, seed=get_seed())(x)
        x = Convolution2D(self.convolutions[-1], 1, padding='same')(x)
        x_bottom = x = BatchNormalization()(x)

        inverse_conv = self.convolutions[::-1]
        inverse_conv = inverse_conv[1:]
        decoded_layers = []
        for i, nbc in enumerate(inverse_conv):
            if not self.attention_guided:
                x = decoder_block(x, connection[i+1], nbc, use_bn=self.decoder_use_bn, spatial_dropout=self.decoder_spatial_dropout)
            else:
                x = decoder_block_guided(x, connection[i + 1], nbc,  iteration=i+1, attention_layer=x_bottom,
                                         use_bn=self.decoder_use_bn, spatial_dropout=self.decoder_spatial_dropout)
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
    network = DualAttentionUnet(input_shape=(1024, 1024, 4), nb_classes=2, deep_supervision=True, input_pyramid=True, attention_guiding=False)
    network.decoder_dropout = 0.1
    network.set_convolutions([8, 16, 32, 64, 128, 128, 256, 256])
    model = network.create()
