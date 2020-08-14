import matplotlib.pyplot as plt
import sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
import os
import cv2
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa


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


def preprocess_wider_face(gt_file, img_dir):
    print("start preprocessing {} data.".format(gt_file.split("_")[-3]))

    with open(gt_file, "r") as f_r:
        num_of_train_imgs = f_r.read().count('jpg')
        img_paths = []

        # loop for each img
        for _ in tqdm(range(num_of_train_imgs)):
            # use strip() to get rid of "\n"
            img_path = f_r.readline().strip()
            img_paths.append(img_path)
            full_img_path = os.path.join(img_dir, img_path)
            im = cv2.imread(full_img_path)
            im_h, im_w, _ = im.shape

            splited_img_path = img_path.split("/")
            dir_name = splited_img_path.[0]
            txt_fname = splited_img_path[1].replace(".jpg", ".txt")

            num_faces = int(f.readline())
            gt_list = []

            # loop for each bbox label
            for _ in range(num_faces):
                xmin, ymin, w, h, _, _, _, _, _, _ = f_r.readline().split()
                # get bbox coordinate values in ratios of the overall image size
                cx = str((int(xmin) + int(w) / 2) / im_w)
                cy = str((int(ymin) + int(h) / 2) / im_h)
                rw = str(w / im_w)
                rh = str(h / im_h)

                class_id = "0"

                label = " ".join([class_id, cx, cy, rw, rh])
                gt_list.append(label)

            gt_str = "\n".join(gt_list)

            # create a label file for each img
            with open(os.path.join(img_dir, dir_name, txt_fname), "w") as f_w:
                f_w.write(gt_str)

    print("preprocessing {} data is done.".format(gt_file.split("_")[-3]))


def get_utkface_np_data(img_dir, img_size):
    # annotation format: [age]_[gender]_[race]_[date&time].jpg
    np_imgs = []
    labels = []

    imgs_list = glob(os.path.join(img_dir, "*.jpg"))
    for img in imgs_list:
        pil_img = Image.open(img)
        resized_pil_img = pil_img.resize((img_size, img_size))
        np_img = np.asarray(resized_pil_img)
        name_split = img.split("_")
        if not len(name_split) == 4:
            raise ValueError("annotation format is not correct for {}"
                             .format(os.basename(img)))
        age, gender, race, _ = name_split
        np_imgs.append(np_img)
        labels.append((age, gender, race))

    np_imgs = np.array(np_imgs)
    labels = np.array(labels)

    return np_imgs, labels


def process_utkface(img):
    img_shape = img.shape
    pad_ratio = 0.7
    pad_size = img_shape[1] * pad_ratio

    img = normalize_inputs(img)
    img = tf.image.random.flip_left_right(img)
    random_angle_to_rotate = tf.random.uniform(shape=(),
                                                minval=-10 * np.pi / 180,
                                                maxval=10 * np.pi / 180)
    img = tfa.image.rotate(img, random_angle_to_rotate)
    img = tf.pad(img, tf.constant([[pad_size, pad_size],
                                   [pad_size, pad_size],
                                   [0, 0]]), "REFLECT")
    img = tf.image.random_crop(img, size=[img_shape[0],
                                          img_shape[1],
                                          img_shape[2]])


def normalize_inputs(X):
    normalized_X = X.astype("float32") / 255.0
    tf.
    return normalized_X


def normalize_age_labels(y_train, y_test):
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    # np.amax returns the max element from the given array
    max_age = np.amax(y_train)
    y_train = y_train / max_age
    y_test = y_test / max_age

    return y_train, y_test, max_age


def denomalize_age_labels(normalized_preds, normalized_labels, max_age):
    denormalized_preds = normalized_preds * max_age
    denormalized_labels = normalized_labels * max_age
    return denormalized_preds, denormalized_labels
