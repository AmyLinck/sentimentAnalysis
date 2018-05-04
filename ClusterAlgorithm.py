from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
The clustering algorithms DBSCAN and K Means using scikit learn
"""

def DBS(data, sentences, epsilon, min_neighbours):
    """
    DBSCAN algorithm
    :param data: vector data
    :param sentences: the actual sentences (non vectorized), used for visualization
    :param epsilon: radius
    :param min_neighbours: number of minimum neighbours for core point
    :return: clustering labels for all sentences (in order)
    """
    dbs = DBSCAN(eps=epsilon, min_samples=min_neighbours).fit(data[:,1:]) #initialize and train model
    labels = dbs.labels_ #get assigned labels

    """
    colouring for visualization
    """
    LABEL_COLOR_MAP = {-1:'y',
                        0: 'b',
                       1: 'g',
                       2:'r',
                       3:'k',
                        4:'c'
                       }

    label_color = [LABEL_COLOR_MAP[l] for l in labels]
    #y_values = dbs.predict(data[:, 1:]) #predict cluster for a datapoint or datapoints

    """
    visualize data points using tSNE
    """
    fig, ax = plt.subplots()
    tsne = TSNE(n_components=2, init='pca', learning_rate=100)
    X_tsne = tsne.fit_transform(data[:, 1:])
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], marker="o", color=label_color)
    """
    Add actual sentences to visualization if desired
    """
    #for xi, yi, pidi in zip(X_tsne[:, 0], X_tsne[:, 1], sentences[:]):
    #    ax.annotate(str(pidi), xy=(xi, yi))

    plt.show()
    return labels

def KMC(data, sentences, clusters):
    """
    K Means Clustering Algorithm
    :param clusters: the number of clusters
    :param data: vector data
    :param sentences: the actual sentences (non vectorized), used for visualization
    :return: clustering labels for all sentences (in order)
    """

    kmc = KMeans(n_clusters=clusters).fit(data[:,1:]) #initialize and train model
    labels = kmc.labels_ #get assigned labels

    """
    colouring for visualization
    """
    LABEL_COLOR_MAP = {0: 'b',
                       1: 'g',
                       2:'r',
                       3:'c',
                       4:'y'
                       }

    label_color = [LABEL_COLOR_MAP[l] for l in labels]
    #y_values = kmc.predict(data[:,1:]) #predict cluster for a datapoint or datapoints
    """
    visualize data points using tSNE
    """
    tsne = TSNE(n_components=2, init='pca', learning_rate=100)
    X_tsne = tsne.fit_transform(data[:,1:])
    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], marker="o", color=label_color)
    """
    Add actual sentences to visualization if desired
    """
    #for xi, yi, pidi in zip(X_tsne[:, 0], X_tsne[:, 1], data[:,0]):
     #   ax.annotate(str(pidi), xy=(xi, yi))
    plt.show()
    return labels


