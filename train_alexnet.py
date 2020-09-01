import matplotlib
# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")

from config import dogs_vs_cats_config as config
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.patchpreprocessor import PatchPreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from hdf5datagenerator import HDF5DatasetGenerator
from model.alexnet import AlexNet 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import json
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to best model weights file")
args = vars(ap.parse_args())

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug, preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 128, aug=aug, preprocessors=[sp, mp, iap], classes=2)

# initialize the optimizer and compile model
print("[INFO] compiling model")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics="accuracy")

# construct the callback to save only the best model to disk based on the validation loss
checkpoint = ModelCheckpoint(args["weights"], monitor="val_loss", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# train the network
model.fit_generator(trainGen.generator(), 
steps_per_epoch=trainGen.numImages//128, 
validation_data=valGen.generator(), 
validation_steps=valGen.numImages//128, 
epochs=75,
max_queue_size=128*2,
callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()