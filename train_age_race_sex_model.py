import os
import utils
from backbones.xception import Xception
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,\
TensorBoard, ReduceLROnPlateau
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizer import Adam
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy


IMG_DIR = "datasets/UTKFace/"
WEIGHT_DIR = "weights/classification_model/"
LOG_DIR = "log/"
IMG_SIZE = 224
NUM_AGE_OUTPUT = 1
NUM_SEX_OUTPUT = 2
NUM_RACE_OUTPUT = 5
EPOCHS = 50
BATCH_SIZE = 32


def create_callbacks(weight_dir, log_dir, monitor, verbose):
    early_stopping = EarlyStopping(monitor=monitor,
                                   patience=10,
                                   verbose=verbose)

    model_checkpoint = ModelCheckpoint(os.path.join(weight_dir,
                                                    "classification_model")
                                       monitor==monitor,
                                       verbose=verbose,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       period=3)

    reduce_lr = ReduceLROnPlateau(monitor=monitor,
                                  factor=0.1,
                                  patience=3,
                                  verbose=verbose)

    logging = TensorBoard(log_dir=log_dir)

    return [early_stopping, model_checkpoint, reduce_lr, logging]


def freeze_layers(model, num_layers_to_freeze):
    for layer in model.layers[:num_layers_to_freeze]:
        layer.trainable = False
    # train batch normalization layers
    if layer.name.startswith("batch_normalization") or layer.name.endswith("bn"):
        layer.trainable = True

    for layer in model.layers[num_layers_to_freeze:]:
        layer.trainable = True

    return model


if __name__ == "__main__":
    np_imgs, labels = utils.get_utkface_np_data(IMG_DIR, IMG_SIZE)
    X_train, X_test, y_train, y_test = utils.train_test_split(np_imgs, labels)
    num_X_train = len(X_train)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(num_X_train)
    train_ds = train_ds.repeat()
    train_ds = train_ds.map(
                    lambda img, label: (utils.process_utkface(img), label)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)


    model = Xception.build(IMG_SIZE)
    freeze_layers(model, 108)
    callbacks_list = create_callbacks(WEIGHT_DIR, LOG_DIR, "val_loss", 1)

    loss_fn = SparseCategoricalCrossentropy()
    optimizer = Adam()

    train_loss = Mean(name="train_loss")
    train_acc = SparseCategoricalAccuracy(name="train_acc")
    val_loss = Mean(name="val_loss")
    val_acc = SparseCategoricalAccuracy(name="val_acc")

    @tf.function
    def train_step(features, labels):
        with tf.GradientTape() as tape:
            preds = model(features, training=True)
            loss = loss_fn(labels, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(loss)
        train_acc(labels, preds)

    @tf.function
    def val_step(features, labels):
        preds = model(features)
        loss = loss_fn(labels, preds)

        val_loss(loss)
        val_acc(labels, preds)
