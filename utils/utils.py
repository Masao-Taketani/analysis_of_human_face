import matplotlib.pyplot as plt
import sklearn.model_selection import train_test_split
import numpy as np


def plot_train_val_history(history):
    plt.figure(figsize=(18, 6))

    plt.plot(history.history["loss"], label="loss", marker="o")
    plt.plot(history.history["val_loss"], label="val_loss", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc = "best")
    plt.grid(color='gray', alpha=0.2)
    plt.show()


def train_test_split(X, y, random_state=0, test_size=0.1):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                                    X,
                                                    y,
                                                    random_state=random_state,
                                                    test_size=test_size)

    return X_train, X_test, y_train, y_test


def normalize_inputs(X_train, X_test):
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    return X_train, X_test


def normalize_age_labels(y_train, y_test):
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    max_age = np.amax(y_train)
    y_train = y_train / max_age
    y_test = y_test / max_age

    return y_train, y_test
