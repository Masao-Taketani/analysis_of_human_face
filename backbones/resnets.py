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


def _bn_relu(inputs):
    norm = BatchNormalization(axis=-1)(inputs)
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

    def f(inputs):
        conv = Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(inputs)
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

    def f(inputs):
        activation = _bn_relu(inputs)
        return Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(inputs, residual):
    """skip cnnection for inputs and residual and it add them by element-wise sum"""

    input_shape = K.int_shape(inputs)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    are_channels_equal = input_shape[-1] == residual_shape[-1]

    shortcut = inputs
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
                          kernel_regularizer=l2(1e-4))(inputs)

    return Add()([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    def f(inputs):
        for i in range(repetitions):
            init_strides = (1, 1)

            if i == 0 and not is_first_layer:
                init_strides = (2, 2)

            inputs = block_function(filters=filters,
                                   init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i==0)
                                   )(inputs)
        return inputs

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3x3 conv block used for resnet 18 and 34."""

    def f(inputs):
        """since the inputs are passed through bn, relu and maxpool if the inputs are
        from first block of first layer, it starts from Conv2D"""
        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filters,
                              kernel_size=(3, 3),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(inputs)
        else:
            conv1 = _bn_relu_conv(filters=filters,
                                  kernel_size=(3, 3),
                                  strides=init_strides)(inputs)

        residual = _bn_relu_conv(filters=filters,
                                 kernel_size=(3, 3))(conv1)
        return _shortcut(inputs, residual)

    return f


def bottleneck_block(filters, init_strides=(1, 1), is_first_block_of_first_layer):
    """bottleneck block used for resnet 50, 101 and 152"""

    def f(inputs):
        if is_first_block_of_first_layer:
            """since the inputs are passed through bn, relu and maxpool if the inputs are
            from first block of first layer, it starts from Conv2D"""
            conv_1_1 = Conv2D(filters=filters,
                              kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(inputs)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters,
                                     kernel_size=(1, 1),
                                     strides=init_strides)(inputs)

        conv_3_3 = _bn_relu_conv(filters=filters,
                                 kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4,
                                 kernel_size=(1, 1))(conv_3_3)
        return _shortcut(inputs, residual)

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


class ResnetBuilder(object):

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        filters = 64

        if len(input_shape) != 3:
            raise Exception("input shape must be 3 dimentions(rows, cols channels).")

        block_fn = _get_block(block_fn)

        inputs = InputLayer(input_shape=input_shape)
        conv1 = _conv_bn_relu(filters=filters,
                              kernel_size=(7, 7),
                              strides=(2, 2))(inputs)
        maxpool = MaxPool2D(pool_size=(3, 3),
                          strides=(2, 2),
                          padding="same")(conv1)
        block = maxpool
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn,
                                    filters=filters,
                                    repetitions=r,
                                    is_first_layer=(i == 0))(block)
            filters *= 2

        # last activation before global avg pooling and the classifier
        last_f_maps = _bn_relu(block)

        last_f_map_shape = K.int_shape(last_f_maps)
        avgpool = AveragePooling2D(pool_size=(last_f_map_shape[1],
                                              last_f_map_shape[2])
                                   strides=(1, 1))(last_f_map_shape)
        flatten = Flatten()(avgpool)
        # classifier block
        dense = Dense(units=num_outputs,
                      kernel_initializer="he_normal",
                      activation="softmax")(flatten)

        model = Model(inputs=inputs, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape,
                                   num_outputs,
                                   basic_block,
                                   [2, 2, 2, 2])
    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape,
                                   num_outputs,
                                   basic_block,
                                   [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape,
                                   num_outputs,
                                   bottleneck_block,
                                   [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape,
                                   num_outputs,
                                   bottleneck_block,
                                   [3, 4, 23, 3])

    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape,
                                   num_outputs,
                                   bottleneck_block,
                                   [3, 8, 36, 3])
