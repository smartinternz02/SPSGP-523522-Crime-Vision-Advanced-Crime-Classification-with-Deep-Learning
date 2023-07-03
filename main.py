# This is a sample Python script.

import os
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from flask import Flask, request, render_template, redirect
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model(r"C:\Users\RIDDHI PHADE\Downloads\crime_clf (1)\crime_clf\crime.h5", compile=False)


# home page
@app.route('/')
def home():
    return render_template('home.html')


# prediction page
@app.route('/prediction')
def prediction():
    return render_template('predict.html')  # Render the prediction template for GET requests


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = load_img(file_path, target_size=(64, 64))
        x = img_to_array(img)  # Converting image into array
        x = np.expand_dims(x, axis=0)  # expanding Dimension
        pred = np.argmax(model.predict(x), axis=1)  # Predicting the higher probability index
        op = ['Fighting', 'Arrest', 'Vandalism', 'Assault', 'Stealing', 'Arson', 'NormalVideos', 'Burglary', 'Explosion', 'Robbery', 'Abuse', 'Shooting', 'Shoplifting', 'RoadAccidents']
        result = 'The predicted output is '+(str(op[pred[0]]))
    return render_template('predict.html', text=result)


"""Running our application"""
if __name__ == "__main__":
    app.run(debug=True)
