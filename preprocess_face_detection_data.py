import os
import cv2
import math
import random

TRAIN_GT_FILE = "wider_face_split/wider_face_train_bbx_gt.txt"
TEST_GT_FILE = "wider_face_split/wider_face_val_bbx_gt.txt"
img_dir = "WIDER_train/images/"

with open(TRAIN_GT_FILE, "r") as f_r:
    num_of_train_imgs = f_r.read().count('jpg')
    img_paths = []

    for _ in range(num_of_train_imgs):
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

        for _ in range(num_faces):
            xmin, ymin, w, h, _,_,_,_,_,_ = f_r.readline().split()
            cx = str((int(xmin) + int(w) / 2) / im_w)
            cy = str((int(ymin) + int(h) / 2) / im_h)

                          
