# The code is based on the following papers and the code.
# paper
#   [1] Deep Residual Learning for Image Recognition
#        which can be found at https://arxiv.org/abs/1512.03385
#   [2] Identity Mappings in Deep Residual Networks
#        https://arxiv.org/abs/1603.05027
# code
#   https://github.com/raghakot/keras-resnet/blob/master/resnet.py


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
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        conv = Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """It is for improved resnets."""

    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """skip cnnection for input and residual and it add them by element-wise sum"""

    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    are_channels_equal = input_shape[-1] == residual_shape[-1]

    shortcut = input
    """check if the sizes of the input_shape and the residual_shape are equal.
    if not, adjust the input size to the resisual size.
    [note] we assume we only do downsampling conv, so input size has to be bigger
    if the input size and the residual size are different. So stride ratios
    (input/residual) has to be greater than 1(>1) even if their sizes are different.
    """
    if stride_width > 1 or stride_height > 1 or not are_channels_equal:
        shortcut = Conv2D(filters=residual_shape[-1],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(1e-4))(input)

    return Add()([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)

            if i == 0 and not is_first_layer:
                init_strides = (2, 2)

            input = block_function(filters=filters,
                                   init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i==0)
                                   )(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3x3 conv block used for resnet 18 and 34."""

    def f(input):
        """since the input is passed through bn, relu and maxpool if the input is
        from first block of first layer, it starts from Conv2D"""
        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filters,
                              kernel_size=(3, 3),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters,
                                  kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters,
                                 kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer):
    """Bottleneck block used for resnet 50, 101 and 152"""

    def f(input):
        if is_first_block_of_first_layer:
            """since the input is passed through bn, relu and maxpool if the input is
            from first block of first layer, it starts from Conv2D"""
            conv_1_1 = Conv2D(filters=filters,
                              kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters,
                                     kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters,
                                 kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4,
                                 kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _get_block(block_fn_name):
    if isinstance(block_fn_name, str):
        # globals(): it returns global variables and funcs in a dict format
        res = globals().get(block_fn_name)
        if not res:
            raise ValueError("block function named {} does not exist.".format(
                                                                block_fn_name))
        return res
    else:
        raise ValueError("block function named has to be string.")
