'''
name: GMM (Gaussian Mixeture Model)

see: scikit-learn

'''
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from collections import Conuter
from models.GenericModel import Model
import numpy as np

class GMM(Model):
    def __init__(self, n_clusters = 30, cov_type='full', iter = 100, snippet_length=75, languages=2):
        '''        
        :param n_clusters: total number of clusters to use for GMM, if different cluster sizes are needed, 
        use [clusters1, clusters2, clusters3...] where the index of the cluster corresponds to the language index
        :param cov_type: type of covariance to use, default 'full'
        :param iter: maximum number of iterations for GMM to perform
        :param snippet_length: number of 10 ms mfcc steps to use for a sample
        :param languages: total models to create depending on each language

        Initialize GMM Model
        '''
        if not type(n_clusters) == int:
            self.gmm = []
            print(n_clusters)
            for model in range(languages):
                self.gmm.append(GaussianMixture(n_components=int(n_clusters[model]), 
                                covariance_type=cov_type,
                                max_iter=iter, 
                                random_state=0))

        else:
            self.gmm = []
            for model in range(languages):
                self.gmm.append(GaussianMixture(n_components=n_clusters,
                                covariance_type=cov_type,
                                max_iter=iter, 
                                random_state=0))

        self.snippet_length = snippet_length
        #initialize cluster array, assignment occurs during training
        self.cluster_distributions = [None] * languages
        self.clusters = n_clusters

    def train(self, training_set, language_idx):
        '''
        :param training_set: samples to train model over
        :param language_idx: language index corresponding to language model
        model assumes each utterance is independent of any other utterance,
        where utterance is a 10ms snippet
        '''
        self.gmm[language_idx] = self.gmm[language_idx].fit(training_set)
        
        self.cluster_distributions[language_idx] = \
            self.make_cluster_dist(training_set, self.gmm[language_idx])

    def make_cluster_dist(self, training_set, gmm):
        '''
        :param training_set: training set for one language vectors for each sample
        :gmm: language specific gmm model used for training
        '''

        cluster_dist = Counter(gmm.predict(training_set))
        cluster_counts = sum(cluster_dist.values())
        for cluster in cluster_dist:
            cluster_dist[cluster] = cluster_dist[cluster] / cluster_counts

        return cluster_dist

    def bayes_predict(self, test_vec):
        
        preds = []
        scores = []

        for gmm in self.gmm:
            preds.append(gmm.predict(test_vec))
            # score = log (P(vec| z, lang))
            scores.append(gmm.score_samples(test_vec))


        # sum(log(P(vec|z,lang)) + log(z|lang))
        prob_sum = []
        for pred,score,dist in zip(preds, scores, self.cluster_distributions):
            prob_sum.append(np.sum(score) + np.sum(np.log([dist[cluster] for cluster in pred])))
        
        return prob_sum

    def predict(self, x):
        '''
        :features: vector representing features of a single instance
        '''
        return np.argmax(self.bayes_predict(x))

    def predict_all(self, x_list):
        pred = []
        for x in x_list:
            pred.append(self.predict(x))
        return pred

    def adjust_data_to_model(self, data):
        '''
        :param data: data to adjust to model expecting len(data)x3x75x13
        :return: tuple of data with adjusted dimensions and it's new shape
        '''
        data_out = []
        for vec in data:
            vec = np.swapaxes(vec,0,1)
            if not len(vec) < self.snippet_length:
                vec = vec.reshape(self.snippet_length, np.shape(vec)[1] * np.shape(vec)[2])
                data_out.append(vec)
        shape = np.shape(data_out)
        return data_out