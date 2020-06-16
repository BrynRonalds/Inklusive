# Script for getting Flask app set-up
from flask import Flask, render_template, request, flash
from flask import send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
# """
# Temporarily including full model prediction here,
# instead of separate .py file or class
#
# * Will need to clean this  up!
# """
# import pandas as pd
# import tensorflow as tf
# from sklearn.neighbors import NearestNeighbors
# from pickle import dump, load
# from inklusive.functions import read_img, resize_img, normalize_img
# from inklusive.config import path_data_cln, tattoodf_name
# tattoo_df = pd.read_csv(tattoodf_name)
#
# # Load the filenames of all training images, and the trained model predictions
# f1 = os.path.join(path_data_cln, 'train_filenames.pkl')
# f2 = os.path.join(path_data_cln, 'train_images.pkl')
# train_filenames = load(open(f1, "rb"))
# train_images = load(open(f2, "rb"))
# # Load the VGG19 model (tansfer learning):
# IMG_SIZE = 256
# shape_img = (IMG_SIZE, IMG_SIZE, 3)
# model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
#                                         input_shape=shape_img)
# output_shape_model = tuple([int(x) for x in model.output.shape[1:]])

### Start actual app stuff:
# Define
UPLOAD_FOLDER = os.path.join(os.getcwd(),'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}


# Create the application object
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
	title_text = 'Inklusive - by Bryn Ronalds'
	return render_template('index.html',
    title=title_text, username='brynron')

# Getting the uploaded image:
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/recommendations', methods=['GET', 'POST'])
def upload_file():
    """Return a rendered web page with the recommended tattoo artists when a user clicks
    `Upload` button.
    Returns:
        A rendered recommendation webpage.
    """
    if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file')
			return redirect(request.url)
		img_file = request.files['file']
		img_folder_path = app.config['UPLOAD_FOLDER']
		if img_file.filename == '':
			flash('No selected file!')
			return redirect(request.url)
		if img_file and allowed_file(img_file.filename):
			filename = secure_filename(img_file.filename)
            img_full_path = os.path.join(img_folder_path, filename)
            img_file.save(os.path.join(img_full_path))
			return redirect(url_for('upload_file',filename=filename))
	return


		# # Now have predict using model
		# imgs_test = read_img(img_full_path)
		# imgs_test_transformed = resize_img(imgs_test,shape_img)
		# imgs_test_transformed = normalize_img(imgs_test_transformed)
		# X_test = np.array(imgs_test_transformed).reshape((-1,) + shape_img)
		# E_test = model.predict(X_test)
		# E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model)))
		# fn = os.path.join(path_data_cln, 'finalized_knn_model.sav')
		# knn = load(open(fn, "rb"))
		# for i, emb_flatten in enumerate(E_test_flatten):
    	# 	_, indices = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
    	# 	imgs_retrieval = [train_images[idx] for idx in indices.flatten()] # retrieval images
    	# 	imgs_path_retrieval = [train_filenames[idx] for idx in indices.flatten()] # retrieve filenames
		# artist = []
		# studio = []
		# for fpath in imgs_path_retrieval:
    	# 	_, filen = os.path.split(fpath)
    	# 	sl_df = tattoo_df[tattoo_df['filename'] == filen]
    	# 	artist.append(sl_df.tail(1)['artist handle'].values[0])
    	# 	studio.append(sl_df.tail(1)['studio name'].values[0])
		# result_images = (imgs_retrieval, artist, studio)
		# outname = 'similartats_for_' + filename + '.png'
		# outfile = os.path.join(os.getcwd(),'static/assets/img',outname)
		# plot_query_retrieval(imgs_test, result_images, outfile, 3)


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
