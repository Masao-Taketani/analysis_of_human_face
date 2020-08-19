import os
import utils
from backbones.xception import Xception
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,\
TensorBoard, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy, \
MeanAbsolutePercentageError
from tensorflow.keras.optimizer import Adam
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

flags.DEFINE_bool("use_xla",
                  False,
                  "Whether to train with XLA.")

flgas.DEFINE_bool("use_mp",
                  False,
                  "Wether to train with mixed precision.")


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


def main(_):
    if FLAGS.use_xla:
        tf.config.optiimzer.set_jit(True)
    if FLAGS.use_mp:

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    np_imgs, labels = utils.get_utkface_np_data(IMG_DIR, IMG_SIZE)
    X_train, X_test, y_train, y_test = utils.train_test_split(np_imgs, labels)
    y_train, max_age = utils.normalize_age_labels(y_train)
    y_test, _ = utils.normalize_age_labels(y_test)
    num_X_train = len(X_train)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # Applying normalization before `ds.cache()` to re-use it.
    # Note: Random transformations (e.g. images augmentations) should be applied
    # after both `ds.cache()` (to avoid caching randomness) and `ds.batch()` (for
    # vectorization [1]).
    train_ds = train_ds.map(
                    lambda img, label: (utils.normalize_inputs(img),
                                        categorize_gendar_and_race(label),
                    num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map()
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
                    lambda img, label: (utils.normalize_inputs(img),
                                        categorize_gendar_and_race(label),
                    num_parallel_calls=AUTOTUNE)
    val_ds = tf.data.Dataset.batch(BATCH_SIZE)

    model = Xception.build(IMG_SIZE)
    freeze_layers(model, 108)
    callbacks_list = create_callbacks(WEIGHT_DIR, LOG_DIR, "val_loss", 1)

    age_loss = MeanAbsolutePercentageError()
    gender_loss = SparseCategoricalCrossentropy()
    race_loss = SparseCategoricalCrossentropy()
    optimizer = Adam()

    train_age_loss = Mean(name="train_age_loss")
    train_gender_loss = Mean(name="train_gender_loss")
    train_race_loss = Mean(name="train_race_loss")
    train_total_loss = Mean(name="train_total_loss")

    val_age_loss = Mean(name="val_age_loss")
    val_gender_loss = Mean(name="val_gender_loss")
    val_race_loss = Mean(name="val_race_loss")
    val_total_loss = Mean(name="val_total_loss")
    #train_acc = CategoricalAccuracy(name="train_acc")
    #val_acc = CategoricalAccuracy(name="val_acc")

    @tf.function
    def train_step(features, labels):
        with tf.GradientTape(persistent=True) as tape:
            preds = model(features, training=True)
            age_loss = age_loss_fn(labels[0], preds[0])
            gender_loss = gender_loss_fn(labels[1], preds[1])
            race_loss = race_loss_fn(labels[2], preds[2])
            total_loss = age_loss + gender_loss + race_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_age_loss(age_loss)
        train_gender_loss(gender_loss)
        train_race_loss(race_loss)
        train_total_loss(total_loss)
        #train_acc(labels, preds)

    @tf.function
    def val_step(features, labels):
        preds = model(features)
        age_loss = age_loss_fn(labels[0], preds[0])
        gender_loss = gender_loss_fn(labels[1], preds[1])
        race_loss = race_loss_fn(labels[2], preds[2])
        total_loss = age_loss + gender_loss + race_loss

        val_age_loss(age_loss)
        val_gender_loss(gender_loss)
        val_race_loss(race_loss)
        val_total_loss(total_loss)
        #val_acc(labels, preds)

    # train and validation loop
    for epoch in range(EPOCHS):
        # reset all of the metrics at the start of each epoch
        train_age_loss.reset_states()
        train_gender_loss.reset_states()
        train_race_loss.reset_states()
        train_total_loss.reset_states()
        val_age_loss.reset_states()
        val_gender_loss.reset_states()
        val_race_loss.reset_states()
        val_total_loss.reset_states()

        for imgs, labels in train_ds:
            train_step(imgs, labels)

        for val_imgs, val_labels in val_ds:
            val_step(val_imgs, val_labels)

        train_log = "EPOCH {}, Age Loss: {}, Gender Loss {}, Race Loss {}, "
                    "Total Loss {} || Val Age Loss: {}, Val Gender Loss {}, "
                    "Val Race Loss {}, Val Total Loss {}"
        print(train_log.format(epoch + 1,
                               train_age_loss.result(),
                               train_gender_loss.result(),
                               train_race_loss.result(),
                               train_total_loss.result(),
                               val_age_loss.result(),
                               val_gender_loss.result(),
                               val_race_loss.result(),
                               val_total_loss.result())


if __name__ == "__main__":
    tf.conpat.v1.app.run()
