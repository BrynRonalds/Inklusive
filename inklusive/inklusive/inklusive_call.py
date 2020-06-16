# Functions/class to run the model on the input images
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pickle import dump, load
from .config import path_models, path_data_cln, tattoodf_name
from .functions import image_transform
import os
import numpy as np

def inklusive_call(test_image, model, top_n):
    """
    Get the top suggested tattoo images based on user upload.
    Want the resulting image filenames, the artist handle and studio name.
    Use pre-trained model to embed the test image, then run NearestNeighbors
    Need to choose which knn to use, "top_n"
    """
    # Load the tattoo dataframe and filenames:
    tattoo_df = pd.read_csv(tattoodf_name)
    train_filenames = load(open(os.path.join(path_data_cln, 'train_filenames.pkl'),'rb'))

    shape_resize = (256, 256, 3)
    img_test_resize = image_transform(test_image, shape_resize)
    X_test = np.array(img_test_resize).reshape((-1,) + shape_resize)

    # Get image embeddings using pre-trained model:
    output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
    E_test = model.predict(X_test)
    E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model)))

    del E_test, X_test, output_shape_model

    # Now load and run the NearestNeighbors models
    knnmodel_name = os.path.join(path_models,'finalized_knn' + str(top_n) +'_model.sav')
    knn = load(open(knnmodel_name, 'rb'))

    imgs_info_df = pd.DataFrame(columns=['path','artist handle','studio handle','studio name'])
    for i, emb_flatten in enumerate(E_test_flatten):
        _, indices = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
        j = 0
        for idx in indices.flatten():
            full_path = train_filenames[idx]
            _, filen = os.path.split(full_path)
            sl_df = tattoo_df[tattoo_df['filename'] == filen]
            artist_handle = sl_df.tail(1)['artist handle'].values[0]
            studio_handle = sl_df.tail(1)['studio handle'].values[0]
            studio_name = sl_df.tail(1)['studio name'].values[0]
            values_to_add = {'path': filen, 'artist handle': artist_handle, 'studio handle': studio_handle, 'studio name': studio_name}
            row_to_add = pd.Series(values_to_add, name=j)
            imgs_info_df = imgs_info_df.append(row_to_add)
            j += 1

    return imgs_info_df
