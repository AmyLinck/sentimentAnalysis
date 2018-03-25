from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import somoclu
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KMC():
    kmc = KMeans

class som():
    som = somoclu

class EM():
    em = GaussianMixture