import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
import random
import math
import cv2
import os
from matplotlib import pyplot as plt
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers import (
    Dense,
    Activation,
    Dropout,
    Flatten,
    Input,
    AveragePooling2D,
    BatchNormalization,
)
from keras.models import Model
from keras.utils import plot_model, np_utils
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    LearningRateScheduler,
)
from time import time
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.utils.generic_utils import get_custom_objects
from keras.regularizers import l2

BATCH_SIZE = 16
# TODO: change class size and species name here
N_CLASSES = 12
species = [
    "Alopias",
    "Carcharias",
    "Carcharodon",
    "Galeocerdo",
    "Heterodontus",
    "Hexanchus",
    "Negaprion",
    "Orectolobus",
    "Prionace",
    "Rhincodon",
    "Sphyrna",
    "Triaenodon",
]

def precision(matrix):
    avg_precise = 0

    for i in range(N_CLASSES):
        tp = matrix[i][i]
        fp = np.sum(matrix[:, i])

        if fp != 0:
            avg_precise += tp / fp

    return avg_precise / N_CLASSES


def recall(matrix):
    avg_recall = 0

    for i in range(N_CLASSES):
        tp = matrix[i][i]
        fn = np.sum(matrix[i, :])
        avg_recall += tp / fn

    return avg_recall / N_CLASSES


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def build_inceptionV3(
    img_shape=(416, 416, 3),
    n_classes=N_CLASSES,
    l2_reg=0.0,
    load_pretrained=True,
    freeze_layers_from="base_model",
):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = "imagenet"
    else:
        weights = None

    # Get base model
    base_model = InceptionV3(
        include_top=False, weights=weights, input_tensor=None, input_shape=img_shape
    )

    # Add final layers
    x = base_model.output
    x = AveragePooling2D((8, 8), strides=(8, 8), name="avg_pool")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="swish", name="dense_1", kernel_initializer="he_uniform")(
        x
    )
    x = Dropout(0.25)(x)
    predictions = Dense(
        n_classes,
        activation="softmax",
        name="predictions",
        kernel_initializer="he_uniform",
    )(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == "base_model":
            print("   Freezing base model layers")
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print("   Freezing from layer 0 to " + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
                layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
                layer.trainable = True

    return model


def build_inception_resnet_V2(
    img_shape=(416, 416, 3),
    n_classes=N_CLASSES,
    l2_reg=0.0,
    load_pretrained=True,
    freeze_layers_from="base_model",
):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = "imagenet"
    else:
        weights = None

    # Get base model
    base_model = InceptionResNetV2(
        include_top=False, weights=weights, input_tensor=None, input_shape=img_shape
    )

    # Add final layers
    x = base_model.output
    x = AveragePooling2D((8, 8), strides=(8, 8), name="avg_pool")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="swish", name="dense_1", kernel_initializer="he_uniform")(
        x
    )
    x = Dropout(0.25)(x)
    predictions = Dense(
        n_classes,
        activation="softmax",
        name="predictions",
        kernel_initializer="he_uniform",
    )(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == "base_model":
            print("   Freezing base model layers")
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print("   Freezing from layer 0 to " + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
                layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
                layer.trainable = True

    return model

if __name__ == "__main__":

    test_X = np.load('X_test.npy')  

    # TODO: change the model you want to analyze here
    #model_final_inception_v2 = build_inception_resnet_V2()
    #model_final_inception_v2.load_weights('inception_resnet_crops.h5')
    model_final_inception_v3 = build_inceptionV3()
    model_final_inception_v3.load_weights('inception_v3_crops+images.h5')

    #y_pred = model_final_inception_v2.predict(test_X,batch_size=BATCH_SIZE,verbose=1)
    y_pred = model_final_inception_v3.predict(test_X,batch_size=BATCH_SIZE,verbose=1)

    y_true = np.load("Y_test.npy")

    ground_truth = []
    preds = []

    for i in range(len(y_true)):
        y_true_arg = np.argmax(y_true[i])
        ground_truth.append(y_true_arg)
        preds_arg = np.argmax(y_pred[i])
        preds.append(preds_arg)

    confusion_matrix = confusion_matrix(ground_truth, preds)
    print(confusion_matrix)

    precise = precision(confusion_matrix)
    recall = recall(confusion_matrix)
    f1 = f1_score(precise, recall)
    #acc = accuracy_score(val_labels, class_predicted) * 100
    print(' ')
	#print("Accuracy for validation set: " + str(acc) + "%")
	print(' ')

	print('precision = ' + str(precise))
	print('recall =' + str(recall))
	print('F1 = ' + str(f1))

    # Plot confusion matrix
    class_list = range(0, N_CLASSES)
    class_names = species
    from plot_confusion_matrix import plot_confusion_matrix

    # Plot non-normalized confusion matrix
    import matplotlib.pyplot as plt
    # TODO: need to update the figure size and name when more data coming in 
    plt.figure(figsize=(13, 10), dpi=80)
    '''plot_confusion_matrix(confusion_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig('V2_unnorm_12.png')'''
    plot_confusion_matrix(confusion_matrix, classes=class_names,
                           title='Confusion matrix, without normalization')
    plt.savefig('V3_unnorm_12.png') 
    plt.show()

    # Plot normalized confusion matrix
    # TODO: need to update the figure size and name when more data coming in
    plt.figure(figsize=(13, 10), dpi=80)
    '''plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('V2_norm_12.png')'''
    plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
                           title='Normalized confusion matrix')
    plt.savefig('V3_norm_12.png') 
    plt.show()
