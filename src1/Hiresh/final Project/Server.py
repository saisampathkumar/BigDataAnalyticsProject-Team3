from flask import request, redirect
from flask import Flask, render_template
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import cv2
from pickle import load
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from Kmeans import KMeans
from extractfeatures import extract_features
from extractfeatures import extract_features_image
from BLEU import isTrain
import cgi, os
import cgitb; cgitb.enable()
form = cgi.FieldStorage()


app = Flask(__name__)
PEOPLE_FOLDER = os.path.join('lab3', 'static')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
@app.route('/')
def signup():
    return render_template('Homepage.html')

@app.route('/pdfreader', methods = ['GET', 'POST'])
def readpdf():
    f = request.files['filename']
    file=""
    if f.filename:
        fn = f.filename
        file=fn
        f.save(os.path.join('static', fn))
    features, sift_features, sift_kmeans = extract_features_image(file,file.split('.')[0])
    caption = isTrain(features, sift_features, sift_kmeans, file.split('.')[0])
    phrase=[]
    phrase.append(caption[1:len(caption)-1])
    return render_template('Homepage.html', core=phrase,user_image="/static/"+ file)

if __name__ == '__main__':
    app.run()