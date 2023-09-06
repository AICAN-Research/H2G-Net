from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.layers import GlobalAveragePooling2D


def get_classifier(network="mobile", input_shape=(512, 512, 3), dense_val=100, nb_classes=2, dropout_val=0.5):
    if network == "inceptionv3":
        input_tensor = Input(shape=input_shape)
        base_model = InceptionV3(include_top=False, weights='imagenet')

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(dense_val, activation='relu')(x)
        x = Dropout(rate=dropout_val)(x)
        x = Dense(nb_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=x)
    elif network == "mobile":
        input_tensor = Input(shape=input_shape)
        base_model = MobileNetV2(include_top=False, weights='imagenet')

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(dense_val, activation='relu')(x)
        x = Dropout(rate=dropout_val)(x)
        x = Dense(nb_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=x)
    else:
        raise ValueError("Unknown network architecture chosen:", network)
    
    return model
