import numpy as np
import umap
import pdb
import umap.plot
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from scipy.optimize import linear_sum_assignment as linear_assignment



class Umap:
    def __init__(self, n_neighbors=15, n_components=2, metric='euclidean', random_state=45):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state

    def transform(self, text_embed, image_embed):
        umap_model_text  = umap.UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components,
                               metric=self.metric, random_state=self.random_state)
        umap_model_image = umap.UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components,
                               metric=self.metric, random_state=self.random_state)
        text_embed = umap_model_text.fit(text_embed)
        image_embed = umap_model_image.fit(image_embed)
        return text_embed, image_embed
    
    def plot_umap(self, text_embed, image_embed, label, text_score, image_score, experiment_num, epoch, proj_label='feat'):
        text_fig = umap.plot.points(text_embed, labels=label)
        text_fig.figure.savefig('EXPERIMENT_{}/umap/{}_{}_eeg_umap_plot_kmean_{}.png'.format(experiment_num, epoch, proj_label, text_score))
        
        image_fig = umap.plot.points(image_embed, labels=label)
        image_fig.figure.savefig('EXPERIMENT_{}/umap/{}_{}_image_umap_plot_kmean_{}.png'.format(experiment_num, epoch, proj_label, image_score))

        
class K_means:
    def __init__(self, n_clusters=39, random_state=45):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def transform(self, text_embed, image_embed, Y_text=None, Y_image=None):
        
        text_label = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init='k-means++').fit_predict(text_embed)
        text_score = self.cluster_acc(Y_text, text_label)

        image_label = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init='k-means++').fit_predict(image_embed)
        image_score = self.cluster_acc(Y_image, image_label)

        return (text_label, image_label), (text_score, image_score)
    
    # Thanks to: https://github.com/k-han/DTC/blob/master/utils/util.py
    def cluster_acc(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size
    


class TsnePlot:
    def __init__(self, perplexity=30, learning_rate=200, n_iter=1000):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
    def plot(self, embedding, labels, score, experiment_num, epoch, proj_label):
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter)
        reduced_embedding = tsne.fit_transform(embedding)
        
        # Create scatter plot with different colors for different labels
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_labels)))
        fig, ax = plt.subplots()
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(reduced_embedding[mask, 0], reduced_embedding[mask, 1], c=colors[i], label=label)
        ax.legend()
        plt.savefig('EXPERIMENT_{}/tsne/{}_{}_eeg_umap_plot_kmean_{}.png'.format(experiment_num, epoch, proj_label, score))

        

