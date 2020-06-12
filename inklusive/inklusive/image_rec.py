# Script for getting image recommendations
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from pickle import dump, load
from .functions import read_img, read_imgs_dir, resize_img, normalize_img
from .config import path_data_cln, path_data_upld, tattoodf_name



# Load the filenames of all training images, and the trained model predictions
f1 = os.path.join(path_data_cln, 'train_filenames.pkl')
f2 = os.path.join(path_data_cln, 'E_train_flatten')
img_filenames = load(open(f1, "rb"))
pred_training = load(open(f2, "rb"))

# Get the user-loaded image
testdir = os.path.join(PROJDIR, 'data/processed/Inklusive_database/test/'
extensions = [".jpg", ".jpeg"]
parallel = True
imgs_test, filen_test = read_imgs_dir(tesdir, extensions, parallel=parallel)

# Load the VGG19 model (tansfer learning):
IMG_SIZE = 256
shape_img = (IMG_SIZE, IMG_SIZE, 3)
model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                        input_shape=shape_img)
output_shape_model = tuple([int(x) for x in model.output.shape[1:]])

# Apply transformations to all images
class ImageTransformer(object):

    def __init__(self, shape_resize):
        self.shape_resize = shape_resize

    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed

transformer = ImageTransformer(shape_img)
imgs_test_transformed = apply_transformer(imgs_test, transformer, parallel=parallel)
X_test = np.array(imgs_test_transformed).reshape((-1,) + shape_img)
E_test = model.predict(X_test)
E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model)))

# Load in the NearestNeighbors model and get image similarities:
fn = os.path.join(DATADIR, 'finalized_knn_model.sav')
knn = load(open(fn, "rb"))
