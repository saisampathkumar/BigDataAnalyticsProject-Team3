from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

class KMeans:
    def cluster_features(img_descs, cluster_model):
        """
        Cluster the training features using the cluster_model
        and convert each set of descriptors in img_descs
        to a Visual Bag of Words histogram.
        Parameters:
        -----------
        X : list of lists of SIFT descriptors (img_descs)
        training_idxs : array/list of integers
            Indicies for the training rows in img_descs
        cluster_model : clustering model (eg KMeans from scikit-learn)
            The model used to cluster the SIFT features
        Returns:
        --------
        X, cluster_model :
            X has K feature columns, each column corresponding to a visual word
            cluster_model has been fit to the training set
        """
        n_clusters = cluster_model.n_clusters
        # Concatenate all descriptors in the training set together
        training_descs = img_descs
        all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
        all_train_descriptors = np.array(all_train_descriptors)

        if all_train_descriptors.shape[1] != 128:
            raise ValueError('Expected SIFT descriptors to have 128 features, got', all_train_descriptors.shape[1])

        # train kmeans or other cluster model on those descriptors selected above
        cluster_model.fit(all_train_descriptors)
        print('done clustering. Using clustering model to generate BoW histograms for each image.')

        # compute set of cluster-reduced words for each image
        img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

        # finally make a histogram of clustered word counts for each image. These are the final features.
        img_bow_hist = np.array(
            [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

        X = img_bow_hist
        print('done generating BoW histograms.')

        return X, cluster_model