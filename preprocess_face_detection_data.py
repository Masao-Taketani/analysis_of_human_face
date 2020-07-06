import os
import cv2
import math
import random
from tqdm import tqdm


TRAIN_GT_FILE = "datasets/WIDER/wider_face_split/wider_face_train_bbx_gt.txt"
VAL_GT_FILE = "datasets/WIDER/wider_face_split/wider_face_val_bbx_gt.txt"
TRAIN_IMG_DIR = "datasets/WIDER/train/imgs/"
VAL_IMG_DIR = "datasets/WIDER/val/imgs/"


def preprocess(gt_file, img_dir):
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
                xmin, ymin, w, h, _,_,_,_,_,_ = f_r.readline().split()
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


if __name__ == "__main__":
    preprocess(TRAIN_GT_FILE, TRAIN_IMG_DIR)
    preprocess(VAL_GT_FILE, VAL_IMG_DIR)
