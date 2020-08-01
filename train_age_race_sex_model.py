import utils


IMG_DIR = ""
IMG_SIZE = 200
NUM_AGE_OUTPUT = 1
NUM_SEX_OUTPUT = 2
NUM_RACE_OUTPUT = 5


np_imgs, ages, genders, races = utils.preprocess_utkface(IMG_DIR, IMG_SIZE)
X_train, X_test, y_train, y_test = utils.train_test_split(np_imgs, )
X_train, X_test = utils.normalize_inputs(X_train, X_test)
