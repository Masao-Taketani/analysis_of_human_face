from glob import glob
import os


def find_jpgs_and_create_txt(set_type, dpath):
	img_paths = []
	dir_fullpath = os.getcwd()

	# recursively get img file paths under `dpath`
	for cur_dir, dirs, files in os.walk(dpath):
		img_paths += glob(os.path.join(cur_dir, '*.jpg'))

	print("the number of imgs:", len(img_paths))
	txt = '\n'.join(img_paths)

	txt_output = ''
	for ipath in txt.split('\n'):
		txt_output += os.path.join(dir_fullpath, ipath) + '\n'

	with open(os.path.join(dpath[:15], set_type + '.txt') , 'w') as f:
		f.write(txt_output)


if __name__ == "__main__":
	find_jpgs_and_create_txt('train', "datasets/WIDER/WIDER_train")
	find_jpgs_and_create_txt('test', "datasets/WIDER/WIDER_val")
