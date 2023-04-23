
import os
import numpy as np
import time
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# keras

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications.resnet import decode_predictions


#Define a flask_app
app  = Flask(__name__)




def model_predict(img_file):

    #first save the image in cwd, then reload with target size in next step
    # Get the filename
    filename = secure_filename(img_file.filename)
    img_path = os.path.join(os.getcwd(),filename)
    img_file.save(img_path)

    # load an image from file
    image = load_img(img_path, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # load the model
    model = ResNet50()
    # predict the probability across all output classes
    t1 = time.time()
    yhat = model.predict(image)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    return label[1], label[2]*100

@app.route('/',methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST','GET'])
def predict():
    if request.method == 'POST':

        # get the file form request
        img_file = request.files['file']

        class_name, prob = model_predict(img_file)

        return str(class_name)
    
    elif request.method == 'GET':
        return render_template('index.html')
    return None

if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)