import cv2
import numpy as np
from os.path import join
from os import listdir
from keras.utils import np_utils

species = [
    "blasti",
    "bonegl",
    "brhkyt",
    "cbrtsh",
    "cmnmyn",
    "gretit",
    "hilpig",
    "himbul",
    "himgri",
    "hsparo",
    "indvul",
    "jglowl",
    "lbicrw",
    "mgprob",
    "rebimg",
    "wcrsrt",
]

datapath = "./"
N_CLASSES = 16  # Number of classes


def gen_data():
    """Generate numpy files for training, validation and
    testing.
    """
    X_train = []
    Y_train = []
    X_valid = []
    Y_valid = []
    X_test = []
    Y_test = []
    count = 0
    for bird_specie in species:

        # Samples Location
        train_data = join(datapath, "train/" + bird_specie)
        val_data = join(datapath, "valid/" + bird_specie)
        test_data = join(datapath, "test/" + bird_specie)

        # Samples Files
        train_files = listdir(train_data)
        valid_files = listdir(val_data)
        test_files = listdir(test_data)

        for img_file in train_files:

            im = join(train_data, img_file)
            img = cv2.imread(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (416, 416))
            X_train.append(img)
            Y_train += [count]

        for img_file in test_files:
            im = join(test_data, img_file)
            img = cv2.imread(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (416, 416))
            X_test.append(img)
            Y_test += [count]

        for img_file in valid_files:
            im = join(val_data, img_file)
            img = cv2.imread(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (416, 416))
            X_valid.append(img)
            Y_valid += [count]

        count += 1

    X_train = np.asarray(X_train).astype("float32")
    X_train /= 255
    Y_train = np.asarray(Y_train)

    X_valid = np.asarray(X_valid).astype("float32")
    X_valid /= 255
    Y_valid = np.asarray(Y_valid)

    X_test = np.asarray(X_test).astype("float32")
    X_test /= 255
    Y_test = np.asarray(Y_test)
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


if __name__ == "__main__":

    x_train, y_train, x_valid, y_valid, x_test, y_test = gen_data()

    y_train = np_utils.to_categorical(y_train, N_CLASSES)
    y_valid = np_utils.to_categorical(y_train, N_CLASSES)
    y_test = np_utils.to_categorical(y_test, N_CLASSES)

    np.save("X_train.npy", x_train)
    np.save("Y_train.npy", y_train)
    np.save("X_valid.npy", x_valid)
    np.save("Y_valid.npy", y_valid)
    np.save("X_test.npy", x_test)
    np.save("Y_test.npy", y_test)
