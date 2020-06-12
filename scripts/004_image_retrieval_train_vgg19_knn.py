# Script version of "004_image_retrieval_train_vgg19_knn"
import os
import numpy as np
import tensorflow as tf
import skimage.io
from skimage.transform import resize
from multiprocessing import Pool
from pickle import dump
from src.CV_transform_utils import apply_transformer, resize_img, normalize_img
from src.CV_plot_utils import plot_query_retrieval
from sklearn.neighbors import NearestNeighbors

# Make paths
proj_dir = '/Users/brynronalds/Insight/proj_dir/'
train_dir = os.path.join(proj_dir,'data/processed/Inklusive_database/train')

# Read images
def read_img(filePath):
    image = skimage.io.imread(filePath, as_gray=False)
    return image

def read_imgs_dir(dirPath, extensions, parallel=True):
    args = [os.path.join(dirPath, filename)
            for filename in os.listdir(dirPath)
            if any(filename.lower().endswith(ext) for ext in extensions)]
    if parallel:
        pool = Pool()
        imgs = pool.map(read_img, args)
        pool.close()
        pool.join()
    else:
        imgs = [read_img(arg) for arg in args]
    return imgs, args
parallel = True
extensions = [".jpg", ".jpeg"]
imgs_train, filenames_train = read_imgs_dir(train_dir, extensions, parallel=parallel)

# Load pre-trained VGG19 model + higher level layers
IMG_SIZE = 256
shape_img_resize = (IMG_SIZE, IMG_SIZE, 3)
model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                        input_shape=shape_img_resize)
output_shape_model = tuple([int(x) for x in model.output.shape[1:]])

# Apply transformations to all images
class ImageTransformer(object):

    def __init__(self, shape_resize):
        self.shape_resize = shape_resize

    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed

transformer = ImageTransformer(shape_img_resize)
imgs_train_transformed = apply_transformer(imgs_train, transformer, parallel=parallel)
X_train = np.array(imgs_train_transformed).reshape((-1,) + shape_img_resize)

# Pickle images and filenames, delete unnecessary variables:
tr_img = os.path.join(train_dir,'train_images.pkl')
tr_fln = os.path.join(train_dir,'train_filenames.pkl')
dump(imgs_train, open(tr_img, 'wb'))
dump(filenames_train, open(tr_fln, 'wb'))

del imgs_train, filenames_train
del transformer, imgs_train_transformed

# Create embeddings using model
print('Embedding model, training data')
E_train = model.predict(X_train)
E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))
preds = os.path.join(train_dir,"E_train_flatten.pkl")
dump(E_train_flatten, open(preds, "wb"))

del E_train, preds, tr_img, tr_fln, IMG_SIZE, output_shape_model

# Fit kNN model on training images
print("Fitting k-nearest-neighbour model on training images...")
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(E_train_flatten)
filename = os.path.join(proj_dir,'models/finalized_knn_model.sav')
dump(knn, open(filename, 'wb'))
