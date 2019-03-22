from glob import glob
import os

def find_all_jpg_and_create_txt(test_set, dir):
	img_paths = []

	for cur_dir, dirs, files in os.walk(dir):
		img_paths += glob(cur_dir + '/*.jpg')

	print(len(img_paths))
	str = '\n'.join(img_paths)

	new_str = ''
	for line in str.split('\n'):
		new_str += '/dl_data/m-taketani/practice/face_detection/data/' + line + '\n'

	with open('/dl_data/m-taketani/practice/face_detection/data/' + test_set + '.txt', 'w') as f:
		f.write(new_str)

	pass


find_all_jpg_and_create_txt('train', "WIDER_train")
find_all_jpg_and_create_txt('test', "WIDER_val")



