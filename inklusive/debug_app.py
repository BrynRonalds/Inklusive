# Script for getting Flask app set-up
from flask import Flask, render_template, request, flash
from flask import send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
"""
Temporarily including full model prediction here,
instead of separate .py file or class

* Will need to clean this  up!
"""
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from pickle import dump, load
from inklusive.functions import read_img, resize_img, normalize_img
from inklusive.config import path_data_cln, tattoodf_name
tattoo_df = pd.read_csv(tattoodf_name)


UPLOAD_FOLDER = os.path.join(os.getcwd(),'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = "inklusive_secret_key"
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
# Put this back into html when can get the recommendations working:
# id="image_upload" action="recommendations"
@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file')
			return redirect(request.url)

		img_file = request.files['file']
		if img_file.filename == '':
			flash('No submitted file')
			return redirect(request.url)

		if img_file and allowed_file(img_file.filename):
			filename = secure_filename(img_file.filename)
			img_file.save(os.path.join(UPLOAD_FOLDER, filename))
			return redirect(url_for('upload_file', filename=filename))
	return


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
