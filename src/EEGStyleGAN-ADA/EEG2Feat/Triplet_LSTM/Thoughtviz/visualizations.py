import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import style
style.use('seaborn')
font = {'family' : 'sans-serif',
        # 'weight' : 'bold',
        'size'   : 35}
matplotlib.rc('font', **font)


class Umap:
    def __init__(self, n_neighbors=15, n_components=2, metric='euclidean', random_state=45):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state

    def plot(self, text_embed, labels, score, exp_type, experiment_num, epoch, proj_type):
        umap_model_text  = umap.UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components,
                               metric=self.metric, random_state=self.random_state)
        text_embed = umap_model_text.fit_transform(text_embed)

        max_val = np.max(text_embed)
        min_val = np.min(text_embed)
        text_embed = (text_embed - min_val)/(max_val - min_val)

        # text_fig = umap.plot.points(text_embed, labels=labels)
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab20b')(np.linspace(0, 1, len(unique_labels)))
        fig, ax = plt.subplots()
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(text_embed[mask, 0], text_embed[mask, 1], c=colors[i], label=label, alpha=0.6)
        # ax.legend(fancybox=True, shadow=True, ncol=1, bbox_to_anchor=(0.88, 0.5))
        ax.legend(fancybox=True, shadow=True, ncol=1)
        plt.tight_layout()
        plt.savefig('EXPERIMENT_{}/{}/umap/{}_{}_eeg_umap_plot_kmean_{}.pdf'.format(experiment_num, exp_type, epoch, proj_type, score), bbox_inches='tight')
        plt.close()
        return text_embed
        
class K_means:
    def __init__(self, n_clusters=40, random_state=45):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def transform(self, embed, gt_labels):
        pred_labels = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(embed)
        score       = self.cluster_acc(gt_labels, pred_labels)
        # image_score = K_means_model.score(image_embed, KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(image_embed))
        return score

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
        
    def plot(self, embedding, labels, score, exp_type, experiment_num, epoch, proj_type):
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter)
        reduced_embedding = tsne.fit_transform(embedding)

        max_val = np.max(reduced_embedding)
        min_val = np.min(reduced_embedding)
        reduced_embedding = (reduced_embedding - min_val)/(max_val - min_val)
        
        # Create scatter plot with different colors for different labels
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab20b')(np.linspace(0, 1, len(unique_labels)))
        fig, ax = plt.subplots()
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(reduced_embedding[mask, 0], reduced_embedding[mask, 1], c=colors[i], label=label, alpha=0.6)
        # ax.legend(fancybox=True, shadow=True, ncol=1, bbox_to_anchor=(0.88, 0.5))
        ax.legend(fancybox=True, shadow=True, ncol=1)
        plt.tight_layout()
        plt.savefig('EXPERIMENT_{}/{}/tsne/{}_{}_eeg_tsne_plot_kmean_{}.pdf'.format(experiment_num, exp_type, epoch, proj_type, score), bbox_inches='tight')
        plt.close()
        return reduced_embedding

    def plot3d(self, embedding, labels, score, exp_type, experiment_num, epoch, proj_type):
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=3, perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter)
        reduced_embedding = tsne.fit_transform(embedding)
        
        max_val = np.max(reduced_embedding)
        min_val = np.min(reduced_embedding)
        # print(max_val, min_val)
        reduced_embedding = (reduced_embedding - min_val)/(max_val - min_val)

        # Create scatter plot with different colors for different labels
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab20b')(np.linspace(0, 1, len(unique_labels)))
        # fig, ax = plt.subplots()
        
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111,projection='3d')
        RADIUS = 5.0  # Control this value.
        # ax.set_xlim3d(0.30, 0.60)
        # ax.set_zlim3d(0.1, 0.6)
        # ax.set_ylim3d(0.30, 0.60)

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(reduced_embedding[mask, 0], reduced_embedding[mask, 1], reduced_embedding[mask, 2], c=colors[i], label=label, alpha=0.6)
        # ax.legend(fancybox=True, shadow=True, ncol=1, bbox_to_anchor=(0.88, 0.5))
        ax.legend(fancybox=True, shadow=True, ncol=1)
        plt.tight_layout()
        plt.savefig('EXPERIMENT_{}/{}/tsne/{}_{}_eeg_tsne3d_plot_kmean_{}.pdf'.format(experiment_num, exp_type, epoch, proj_type, score), bbox_inches='tight')
        plt.close()
        return reduced_embedding


        
def save_image(spectrogram, gt, experiment_num, epoch, folder_label):
    # Assuming `spectrogram` is the 3D tensor of shape `(440, 33, 9)`
    num_rows = 2
    num_cols = 2
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 10))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i*num_cols + j
            if index < spectrogram.shape[0]:
                # Get the spectrogram and convert it to a numpy array
                spec = np.squeeze(spectrogram[index].numpy(), axis=0)
                # Plot the spectrogram using a heatmap with the 'viridis' color map
                im = axes[i,j].imshow(spec, cmap='viridis', aspect='auto')

                # Set the title and axis labels
                axes[i,j].set_title('EEG {}'.format(index+1))
                axes[i,j].set_xlabel('Time')
                axes[i,j].set_ylabel('Amplitude')

                # Add colorbar
                # cax = plt.axes([0.95, 0.1, 0.03, 0.8])
                # fig.colorbar(im, cax=cax)

    plt.tight_layout()
    # plt.show()
    plt.savefig('EXPERIMENT_{}/{}/{}_pred.png'.format(experiment_num, folder_label, epoch))
    
    # plt.clf()

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 10))
    spectrogram = gt

    for i in range(num_rows):
        for j in range(num_cols):
            index = i*num_cols + j
            if index < spectrogram.shape[0]:
                # Get the spectrogram and convert it to a numpy array
                spec = np.squeeze(spectrogram[index].numpy(), axis=0)

                # Plot the spectrogram using a heatmap with the 'viridis' color map
                im = axes[i,j].imshow(spec, cmap='viridis', aspect='auto')

                # Set the title and axis labels
                axes[i,j].set_title('EEG {}'.format(index+1))
                axes[i,j].set_xlabel('Time')
                axes[i,j].set_ylabel('Amplitude')

                # Add colorbar
                # cax = plt.axes([0.95, 0.1, 0.03, 0.8])
                # fig.colorbar(im, cax=cax)

    plt.tight_layout()
    # plt.show()
    plt.savefig('EXPERIMENT_{}/{}/{}_gt.png'.format(experiment_num, folder_label, epoch))
    plt.close('all')