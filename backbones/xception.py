# The code is based on the following papers and the code.
# paper
#   [1] Xception: Deep Learning with Depthwise Separable Convolutions
#        which can be found at https://arxiv.org/abs/1610.02357
# code
#   https://github.com/yanchummar/xception-keras/blob/master/xception_model.py


from tensorflow.keras.model import Model
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization, \
Activation, Conv2D, SeparableConv2D, MaxPool2D, GlobalAveragePooling2D, \
GlobalMaxPooling2D
from tensorflow.utils import get_file


WEIGHTS_PATH = ""


def Xception():

    IMG_SIZE = 299
    CLASS_NUM = 1000
    NUM_CHANNEL = 3
    input_shape = (IMG_SIZE, IMG_SIZE, NUM_CHANNEL)

    inputs = InputLayer(shape=input_shape)

    ## Entry Flow
    # Block1
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(inputs)
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

    shortcut3 = Conv2D(728,
                       (1, 1),
                       strides=(2, 2),
                       padding="same",
                       use_bias=False)(x)
    shortcut3 = BatchNormalization()(shortcut3)

    # Block4
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)
    # merge3
    x = Add()([x, shortcut3])
    # At the end the entry flow, we get 19x19x728 feature maps

    ## Middle Flow
    for i in range(8):
        # Block5-12
        shortcut4 = x
        x = Activation("relu")(x)
        x = SeparableConv2D(728, (3, 3), padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(728, (3, 3), padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(728, (3, 3), padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        # merge4-11
        x = Add()([x, shortcut4])
    # At the end the middle flow, we get 19x19x728 feature maps

    ## Exit Flow
    shortcut5 = Conv2D(1024,
                       (1, 1),
                       strides=(2, 2),
                       padding="same",
                       use_bias=False)(x)

    # Block13
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(1024, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)
    # merge5
    x = Add()([x, shortcut5])

    # Block14
    x = SeparableConv2D(1536, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(2048, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)
    # The feature map size after the Global AVG pooling is 2048
    x = Dense(CLASS_NUM, activation="softmax")(x)

    model = Model(inputs, x, name="xception")

    weights_path = get_file("xception_weights.h5",
                            WEIGHTS_PATH,
                            cache_subdir="models")

    model.load_weights(weights_path)

    return model
