# Script for getting Flask app set-up
from flask import Flask, render_template, request, flash
from flask import send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
from base64 import b64encode
from inklusive.config import path_data_cln, path_tattoodf
from inklusive.inklusive_call import inklusive_call
import pickle
import tensorflow as tf

# Load the images & pre-trained model:
IMG_SIZE = 256
shape_resize = (IMG_SIZE, IMG_SIZE, 3)
model_vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                        input_shape=shape_resize)

UPLOAD_FOLDER = os.path.join(os.getcwd(),'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
#app.secret_key = "inklusive_secret_key"
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

@app.route('/recommendations', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file')
			return redirect(request.url)

		file = request.files['file']
		if file.filename == '':
			flash('No submitted file')
			return redirect(request.url)

		if file and allowed_file(file.filename):
			img_folder_path = app.config['UPLOAD_FOLDER']
			filen_test_img = secure_filename(file.filename)
			img_full_path = os.path.join(img_folder_path, filen_test_img)
			file.save(os.path.join(img_full_path))

			encoded = b64encode(file.read()) # encode the binary
			mime = "image/jpg"
			in_img = "data:%s;base64,%s" %(mime, encoded.decode()) # remember to decode the encoded data
			# in_img = send_from_directory(img_folder_path,filen_test_img, as_attachment=True)

		# Return the dataframe of suggested image_transform
		# Columns = filename, artist handle, studio handle, studio name
		tattoos_info_df = inklusive_call(img_full_path, model_vgg19, 8)
		return render_template("results.html", flag="1", input_pic=in_img, output_df=tattoos_info_df)

	#else:
	#	render_template("index.html", flag="0", sel_input=selection, sel_form_result="Empty")
@app.route('/<path:filename>')
def download_file(filename):
	return send_from_directory(path_tattoodf, filename, as_attachment=True)

@app.route('/<path:filename>')
def download_file_input(filename):
	return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
