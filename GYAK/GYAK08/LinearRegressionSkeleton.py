import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epochs = epochs
        self.lr = lr

    def evaluate(self,y_pred,y_test):
        print("Mean Squared Error:", np.mean((y_pred - y_test)**2))

    def fit(self, X_train: np.array, y_train: np.array):
        self.m = 0
        self.c = 0
        self.n = float(len(X_train)) # Number of elements in X

        # Performing Gradient Descent 
        losses = []
        for i in range(self.epochs): 
            y_pred = self.m*X_train + self.c  # The current predicted value of Y

            residuals = y_train - y_pred
            loss = np.sum(residuals ** 2)
            losses.append(loss)
            D_m = (-2/self.n) * sum(X_train * residuals)  # Derivative wrt m
            D_c = (-2/self.n) * sum(residuals)  # Derivative wrt c
            self.m = self.m - self.lr * D_m  # Update m
            self.c = self.c - self.lr * D_c  # Update c
            #if i % 100 == 0:
            #    print(np.mean(y_train-y_pred))

    def predict(self, X):
        return self.m*X + self.c