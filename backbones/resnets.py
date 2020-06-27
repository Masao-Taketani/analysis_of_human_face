from tensorflow.keras.models import Model
from tensorflow.keras.layers import InputLayer, Activation, Dense, Flatten, \
Conv2D, MaxPool2D, AveragePooling2D, Add, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def _bn_relu(input):
    norm = BatchNormalization(axis=-1)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    # dict.setdefault("key", value))
    # if the specified key already exists, it doesn't update the value.
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f
