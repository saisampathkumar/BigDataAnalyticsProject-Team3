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
from numpy import reshape,array

def extract_features(directory):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = dict()
    sift_features = dict()
    sift_kmeans=[]
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature

        # Extracting SIFT features
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        sift_features[image_id] = des
        sift_kmeans.append(des)
        print('>%s' % name)
    return features,sift_features,sift_kmeans

# extract features from all images
directory = '/home/vthotigar/bda_data/image_data'
features,sift_features,sift_kmeans = extract_features(directory)
print('Extracted Features: %d' % len(features))
cluster_model = MiniBatchKMeans(n_clusters=64)
sift_kmeans, cluster_model = KMeans.cluster_features(sift_kmeans, cluster_model)
sift = dict()
j=0
for k, i in features.items():
    sift[k] = sift_kmeans[j]
    j=j+1
print(j)
# save to file
dump(features, open('features.pkl', 'wb'))
dump(sift, open('sift_features.pkl', 'wb'))
dump(cluster_model, open('clustermodel.pkl','wb'))
# sift=dict()
# all_features = load(open('features.pkl', 'rb'))
# sift_features = load(open('sift_features.pkl', 'rb'))
# for k,i in enumerate(all_features):
#     sift[i]=sift_features[k]
#     print(array(sift_features[k]).shape)
# dump(sift, open('sift_features.pkl', 'wb'))
