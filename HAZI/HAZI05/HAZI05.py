import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from typing import Tuple
import seaborn as sns

csv_path="diabetes.csv"

class KNNClassifier:
    
    @property
    def k_neighbors(self):
        return self.k

    def __init__(self, k:int, test_split_ratio :float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio
    
    @staticmethod 
    def load_csv(csv_path:str)-> Tuple[pd.DataFrame,pd.Series]:
        dataset = pd.read_csv(csv_path)
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x,y = dataset.iloc[:,:-1],dataset.iloc[:,-1]
        return x,y
    
    def train_test_split(self, features: pd.DataFrame,
                     labels: pd.DataFrame) -> None:
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        x_train, y_train = features.iloc[:train_size, :].reset_index(drop=True), labels.iloc[:train_size].reset_index(drop=True)
        x_test, y_test = features.iloc[train_size:, :].reset_index(drop=True), labels.iloc[train_size:].reset_index(drop=True)
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def euclidean(self, element_of_x: pd.Series) -> pd.DataFrame:
        return pd.np.sqrt(((self.x_train - element_of_x)**2).sum(axis=1))
    
    def predict(self, x_test: pd.DataFrame) -> pd.DataFrame:
        labels_pred =[]
        for i, x_test_element in x_test.iterrows():

            distances = self.euclidean(x_test_element)
            distances = pd.DataFrame({'distances': distances, 'labels': self.y_train})
            distances.sort_values(by='distances', inplace=True)

            label_pred = mode(distances.iloc[:self.k, 1], keepdims = False).mode

            labels_pred.append(label_pred)
        self.y_preds = pd.Series(labels_pred)
    
    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100
    
    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.y_test, self.y_preds)
    
    @staticmethod
    def best_k(x: pd.DataFrame, y: pd.Series) -> Tuple[int, float]:
        best = (0, 0)
        for i in range(1, 21):
            classifier = KNNClassifier(i, 0.2)
            classifier.train_test_split(x, y)
            classifier.predict(classifier.x_test)
            accuracy = classifier.accuracy()
            if best[1] < accuracy:
                best = (i, round(accuracy, 2))
        return best
