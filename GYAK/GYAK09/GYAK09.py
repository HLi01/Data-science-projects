# %% [markdown]
# ### SVM
# SVM offers very high accuracy compared to other classifiers such as logistic regression, and decision trees. It is known for its kernel trick to handle nonlinear input spaces. It is used in a variety of applications such as face detection, intrusion detection, classification of emails, news articles and web pages, classification of genes, and handwriting recognition.

# %% [markdown]
# Generally, Support Vector Machines is considered to be a classification approach, it but can be employed in both types of classification and regression problems. It can easily handle multiple continuous and categorical variables. SVM constructs a hyperplane in multidimensional space to separate different classes. SVM generates optimal hyperplane in an iterative manner, which is used to minimize an error. The core idea of SVM is to find a maximum marginal hyperplane(MMH) that best divides the dataset into classes.

# %% [markdown]
# ajánlott irodalom: [here](https://towardsdatascience.com/support-vector-machine-explained-8bfef2f17e71), [here](https://monkeylearn.com/blog/introduction-to-support-vector-machines-svm/)

# %%
from IPython.display import Image
Image("img/svm.png",width=300)

# %% [markdown]
# #### Support Vectors
# Support vectors are the data points, which are closest to the hyperplane. These points will define the separating line better by calculating margins. These points are more relevant to the construction of the classifier.
# 
# #### Hyperplane
# A hyperplane is a decision plane which separates between a set of objects having different class memberships.
# 
# #### Margin
# A margin is a gap between the two lines on the closest class points. This is calculated as the perpendicular distance from the line to support vectors or closest points. If the margin is larger in between the classes, then it is considered a good margin, a smaller margin is a bad margin.

# %% [markdown]
# #### How does SVM work?
# The main objective is to segregate the given dataset in the best possible way. The distance between the either nearest points is known as the margin. The objective is to select a hyperplane with the maximum possible margin between support vectors in the given dataset. SVM searches for the maximum marginal hyperplane in the following steps:
# 
# 1. Generate hyperplanes which segregates the classes in the best way. Left-hand side figure showing three hyperplanes black, blue and orange. Here, the blue and orange have higher classification error, but the black is separating the two classes correctly.
# 
# 2. Select the right hyperplane with the maximum segregation from the either nearest data points as shown in the right-hand side figure.
# 
# 

# %%
Image("img/svm2.png",width=600)

# %% [markdown]
# #### Dealing with non-linear and inseparable planes
# Some problems can’t be solved using linear hyperplane, as shown in the figure below (left-hand side).
# 
# In such situation, SVM uses a kernel trick to transform the input space to a higher dimensional space as shown on the right. The data points are plotted on the x-axis and z-axis (Z is the squared sum of both x and y: z=x^2+y^2). Now you can easily segregate these points using linear separation.

# %%
Image("img/svm3.png",width=600)

# %% [markdown]
# ### Exercise

# %%
# Load the built-in breast_cancer dataset: cancer
# NOTE: from sklearn datasets (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()

# %%
# print the names of the 13 features
print(data.feature_names)
# print the label type of cancer
print(data.target_names)


# %%
# print the cancer data (top 3 records)
data.target[:3]

# %%
# print the cancer labels (0:malignant, 1:benign)
print(data.target)

# %%
# Split dataset into training set and test set (sklearn train_test_split)
from sklearn.model_selection import train_test_split
X=data.data
y=data.target
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=24, test_size=0.25, shuffle=True)

# %%
# Create a svm Classifier: clf (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
# Use linear kernel
from sklearn.svm import SVC
clf = SVC(kernel='linear', gamma='auto')

# Train (fit) the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
print(clf.predict(X_test))

# %%
# Model Accuracy: how often is the classifier correct?

print('Train accuracy: ', clf.score(X_train, y_train))
print('Test accuracy: ', clf.score(X_test, y_test))

# %% [markdown]
# # Model Precision: what percentage of positive predictions are truly positive?
# 
# 
# # Model Recall: what percentage of positive datapoints are labelled as such?
# 

# %% [markdown]
# #### Tuning Hyperparameters
# * **Kernel:** The main function of the kernel is to transform the given dataset input data into the required form. There are various types of functions such as linear, polynomial, and radial basis function (RBF). Polynomial and RBF are useful for non-linear hyperplane. Polynomial and RBF kernels compute the separation line in the higher dimension. In some of the applications, it is suggested to use a more complex kernel to separate the classes that are curved or nonlinear. This transformation can lead to more accurate classifiers.
# * **Regularization:** Regularization parameter in python's Scikit-learn C parameter used to maintain regularization. Here C is the penalty parameter, which represents misclassification or error term. The misclassification or error term tells the SVM optimization how much error is bearable. This is how you can control the trade-off between decision boundary and misclassification term. A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.
# * **Gamma:** A lower value of Gamma will loosely fit the training dataset, whereas a higher value of gamma will exactly fit the training dataset, which causes over-fitting. In other words, you can say a low value of gamma considers only nearby points in calculating the separation line, while the a value of gamma considers all the data points in the calculation of the separation line.

# %%
from sklearn.model_selection import GridSearchCV
from sklearn import svm
#(https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
'kernel': ['rbf']}


# Check how the function GridSearchCV works
# Use it with estimator SVC and with the above-given param_grid
# Set the verbose parameter to at least 1
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid, verbose=1)

# Fit the created grid model on your train data
clf.fit(data.data, data.target)

# %%
# Print best parameter after tuning 
# (your created grid model has a function named best_params_)
print(clf.best_params_)

# Print how our model looks after hyper-parameter tuning
# (check the best_estimator_ function)
print(clf.best_estimator_)

# %%
# Predict with the help of your new model: grid_predictions
# As usual, this model also has a 'predict' function
grid_predictions = clf.predict(X_test)



# %%
# Evaluate your model: print its accuracy, precision and recall values
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
print("accuracy: ",clf.best_score_)
print("precision: ", precision_score(grid_predictions, y_test))
print("recall values: ",recall_score(grid_predictions, y_test))


