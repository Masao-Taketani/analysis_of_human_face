import matplotlib.pyplot as plt
import sklearn.model_selection import train_test_split
import numpy as np


def plot_train_val_history(history):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    # accuracy plot
    plot_line_chart(history)
    # loss plot
    plot_line_chart(history, is_acc=False)
    plt.show()


def plot_line_chart(history, is_acc=True):
    if is_acc:
        label = "acc"
    else:
        label = "loss"
    plt.plot(history.history[label], label=label, marker = "o")
    plt.plot(history.history["val_" + label], label="val_" + label, marker="o")
    plt.xlabel("epoch")
    plt.ylabel(label)
    plt.legend(loc = "best")
    plt.grid(color="gray", alpha=0.2)


def plot_result_imgs(pred_vals, labels, num_imgs_to_show=30):
    plt.figure((16, 6))
    num_cols = 10
    for i in range(num_imgs_to_show):
        plt.subplot(num_imgs_to_show // num_cols, num_cols, i+1)
        plt.axis("off")
        pred_val = round(pred_vals[i][0])
        label = round(labels[i])

        # check if the error rate is less than 20% or not
        if abs(pred_val - label) / label < 0.2:
            put_title_on_result_img(label, pred_val)
        else:
            put_title_on_result_img(label, pred_val, is_true=False)


def put_title_on_result_img(label, pred_val, is_true=True):
    if is_true:
        color = "black"
    else:
        color = "red"
    plt.title("label:{}\\npred:{}".format(str(label), str(pred_val)), color=color)


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

    return y_train, y_test, max_age


def denomalize_age_labels(normalized_preds, normalized_labels, max_age):
    denormalized_preds = normalized_preds * max_age
    denormalized_labels = normalized_labels * max_age
    return denormalized_preds, denormalized_labels
