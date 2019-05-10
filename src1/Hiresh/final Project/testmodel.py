from extractfeatures import extract_features_image
from BLEU import isTrain

features, sift_features, sift_kmeans=extract_features_image('C:/Users/Hiresh/Desktop/UMKC/BDAA/Lab 2/New Code/lab3/lab3/Dataset/flickr8k_images/398662202_97e5819b79.jpg','398662202_97e5819b79')
print(isTrain(features,sift_features,sift_kmeans,'398662202_97e5819b79'))