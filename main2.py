import pandas as pd
import numpy as np
from sklearn import preprocessing
from planar_utils import sigmoid
from readFile import read_csv
min_max_scaler = preprocessing.MinMaxScaler()

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    # A1 = np.tanh(Z1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2 
    A2 = sigmoid(Z2)
    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    m = Y.shape[1] # number of example
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), (1-Y))
    cost = -(1.0/m)*np.sum(logprobs)
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
    assert(isinstance(cost, float))
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2 - Y
    dW2 = 1.0/m*np.dot(dZ2, A1.T)
    db2 = 1.0/m*np.sum(dZ2, axis=1, keepdims=True)
    # dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1, 2))
    dZ1 = np.dot(W2.T, dZ2)*A1*(1-A1)
    dW1 = 1.0/m*np.dot(dZ1, X.T)
    db1 = 1.0/m*np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    print('n_x', n_x)
    print('n_y', n_y)
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    dataframe = pd.DataFrame({'A2': A2.tolist()[0]})
    dataframe.to_csv("A2.csv",index=False,sep=',')

    predictions = 1.0*(A2 > 0.5)

    return predictions

# 读取训练数据和预测数据
X_assess, Y_assess, X_test_assess, Y_test_assess = read_csv()

# 数据归一化
X_assess = np.array(X_assess)
X_assess_minmax = min_max_scaler.fit_transform(X_assess)
X_test_assess = np.array(X_test_assess)
X_test_assess_minmax = min_max_scaler.fit_transform(X_test_assess)
Y_assess = np.array(Y_assess)
Y_test_assess = np.array(Y_test_assess)

print('X_assess_minmax', X_assess_minmax)
print('X_test_assess_minmax', X_test_assess_minmax)
# 模型训练开始
# parameters = nn_model(X_assess_minmax.T, Y_assess.T, 1, num_iterations=3000, print_cost=True)
# predictions = predict(parameters, X_test_assess_minmax.T)
# print(predictions.tolist()[0])
# dataframe = pd.DataFrame({'predictions': predictions.tolist()[0]})
# dataframe.to_csv("test.csv",index=False,sep=',')
# print(float((np.dot(Y_test_assess.T, predictions.T) + np.dot((1-Y_test_assess).T, (1-predictions).T)))
# finalResult = (np.dot(Y_test_assess.T, predictions.T) + np.dot((1-Y_test_assess).T,(1-predictions).T))/float(Y_test_assess.size)*100
# print(finalResult[0][0])
# print ('Accuracy %.2f' % float(finalResult) + '%')

hidden_layer_sizes = [50]
for i, n_h in enumerate(hidden_layer_sizes):
    parameters = nn_model(X_assess_minmax.T, Y_assess.T, n_h, num_iterations=5000, print_cost=True)
    predictions = predict(parameters, X_test_assess_minmax.T)
    accuracy = float((np.dot(Y_test_assess.T, predictions.T) + np.dot((1-Y_test_assess).T,(1-predictions).T))/float(Y_test_assess.size)*100)
    print ("Accuracy for {} hidden units: {}%".format(n_h, accuracy))
    fo = open("result2.txt",  mode='a')
    str = "Accuracy for {} hidden units: {}%".format(n_h, accuracy)
    fo.write( str )
    fo.write('\n')