import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
from sklearn.utils import Bunch
from typing import Tuple

class KMeansOnDigits():

    def __init__(self, n_clusters, random_state) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state

    def load_dataset(self) -> None:
        self.digits = load_digits(as_frame=True)

    def predict(self) -> None:
        kmeans = KMeans(self.n_clusters, random_state=self.random_state)
        self.clusters = kmeans.fit_predict(self.digits["data"])

    def get_labels(self) -> None:
        result = np.empty(shape=self.clusters.shape)

        for i in self.digits.target_names:
            mask = (self.clusters == i)
            result[mask] = mode(self.digits.target[mask], keepdims=False).mode
        
        self.labels = result

    def calc_accuracy(self) -> None:
        self.accuracy = accuracy_score(self.digits["target"], self.labels).round(2)

    def confusion_matrix(self) -> None:
        self.mat = confusion_matrix(self.digits["target"], self.labels, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def kmeans_pipeline(self) -> np.ndarray:
        self.load_dataset()
        self.predict()
        self.get_labels()
        self.calc_accuracy()
        self.confusion_matrix()
        return self.mat