import numpy as np
import pandas as pd
import sklearn as sklearn
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

def load_iris_data() -> sklearn.utils.Bunch:
    iris = load_iris()
    return iris

def check_data(iris) -> pd.core.frame.DataFrame:
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    return df.head(5)

def linear_train_data(iris) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    X = df[['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
    y = df['sepal length (cm)'].values
    return X, y

def logistic_train_data(iris) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    X = df.loc[np.where(iris.target < 2)].values
    y = iris.target[np.where(iris.target < 2)]
    return X, y

def split_data(X, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_linear_regression(X_train, y_train) -> sklearn.linear_model._base.LinearRegression:
    return LinearRegression().fit(X_train, y_train)

def train_logistic_regression(X_train, y_train) -> sklearn.linear_model._logistic.LogisticRegression:
    return LogisticRegression(solver='liblinear', random_state=42).fit(X_train, y_train)

def predict(model, X_test) -> np.ndarray:
    return model.predict(X_test)

def plot_actual_vs_predicted(y_test, y_pred) -> plt.Figure:
    fig, ax = plt.subplots()    
    ax.set_title('Actual vs Predicted Target Values')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.scatter(y_test, y_pred)
    return fig

def evaluate_model(y_test, y_pred) -> float:
    return mean_squared_error(y_test, y_pred)