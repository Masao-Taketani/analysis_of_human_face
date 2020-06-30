from tensorflow.keras.model import Model
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization, \
Activation, Conv2D, SeparableConv2D, MaxPool2D, GlobalAveragePooling2D, \
GlobalMaxPooling2D
from tensorflow.utils import get_file


WEIGHT_PATH = ""


def Xception():
    IMG_SIZE = 299
    NUM_CHANNEL = 3
    input_shape = (IMG_SIZE, IMG_SIZE, NUM_CHANNEL)

    img_input = InputLayer(shape=input_shape)

    # Block1
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_inpug)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    shortcut1 = Conv2D(128,
                       (1, 1),
                       strides=(2, 2),
                       padding="same",
                       use_bias=False)(x)
    shortcut1 = BatchNormalization()(shortcut1)

    # Block2
    x = SeparableConv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)

    # merge1
    x = Add()([x, shortcut1])

    shortcut2 = Conv2D(256,
                       (1, 1),
                       strides=(2, 2),
                       padding="same",
                       use_bias=False)(x)
    shortcut2 = BatchNormalization()(shortcut2)

    # Block3
    x = Activation("relu")(x)
    x = SeparableConv2D(256, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(256, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)

    # merge2
    x = Add()([x, shortcut2])
