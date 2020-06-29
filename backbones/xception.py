from tensorflow.keras.model import Model
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization, \
Activation, Conv2D, SeparableConv2D, MaxPool2D, GlobalAveragePooling2D, \
GlobalMaxPooling2D
from tensorflow.utils import get_file


WEIGHT_PATH = ""


def Xception():
