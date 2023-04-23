from LinearRegressionSkeleton import LinearRegression
import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

# %%
model = LinearRegression()
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
X = df['petal width (cm)'].values
y = df['sepal length (cm)'].values
model.fit(X, y)

y_pred = model.predict(X)
