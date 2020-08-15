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


def main():
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    np_imgs, labels = utils.get_utkface_np_data(IMG_DIR, IMG_SIZE)
    X_train, X_test, y_train, y_test = utils.train_test_split(np_imgs, labels)
    num_X_train = len(X_train)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # Applying normalization before `ds.cache()` to re-use it.
    # Note: Random transformations (e.g. images augmentations) should be applied
    # after both `ds.cache()` (to avoid caching randomness) and `ds.batch()` (for
    # vectorization [1]).
    train_ds = train_ds.map(
                    lambda img, label: (utils.normalize_inputs(img), label),
                    num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size=num_X_train)
    #train_ds = train_ds.repeat()
    # Batch after shuffling to get unique batches at each epoch.
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.map(
                    lambda img, label: (utils.process_utkface(img), label),
                    num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_ds = val_ds.map(
                    lambda img, label: (utils.normalize_inputs(img), label),
                    num_parallel_calls=AUTOTUNE)
    val_ds = tf.data.Dataset.batch(BATCH_SIZE)

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

    # train and validation loop
    for epoch in range(EPOCHS):
        # reset all of the metrics at the start of each epoch
        train_loss.reset_states()
        train_acc.reset_states()
        val_loss.reset_states()
        val_acc.reset_states()

        for imgs, labels in train_ds:
            train_step(imgs, labels)

        for val_imgs, val_labels in val_ds:
            val_step(val_imgs, val_labels)

        train_log = "EPOCH {}, Loss: {}, Acc: {}, Val Loss: {}, Val Acc: {}"
        print(train_log.format(epoch + 1,
                               train_loss.result(),
                               train_acc.result() * 100,
                               val_loss.result(),
                               val_acc.result() * 100)


if __name__ == "__main__":
    main()
