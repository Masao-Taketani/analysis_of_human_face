from glob import glob
import os

def find_all_jpg_and_create_txt(test_set, dir):
	img_paths = []

	for cur_dir, dirs, files in os.walk(dir):
		img_paths += glob(cur_dir + '/*.jpg')

	print(len(img_paths))
	str = '\n'.join(img_paths)

	with open(test_set + '.txt', 'w') as f:
		f.write(str)

	pass


find_all_jpg_and_create_txt('train', "WIDER_train")
find_all_jpg_and_create_txt('test', "WIDER_val")



