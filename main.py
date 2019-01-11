# Package imports
import numpy as np
import matplotlib.pyplot as plt
#from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
X,Y=load_planar_dataset()
plt.scatter(X[0, :], X[1, :],c=Y.reshape(X[0,:].shape),  s=40, cmap = plt.cm.Spectral)
plt.show()

shape_X=X.shape
shape_Y=Y.shape
m=X.shape[1]
print("the shape of X is:"+str(shape_X))
print("the shape of Y is:"+str(shape_Y))
print("the training examples:" +str(m))
#np.random.seed(1) # set a seed so that the results are consistent

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float(
    (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
       '% ' + "(percentage of correctly labelled datapoints)")