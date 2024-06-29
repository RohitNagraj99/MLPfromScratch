"""
Applying MLP on Synthetic Dataset
"""

"""
Author Details:

R Anirudh     (1MS17IS084)

Rohit P N     (1MS17IS094)

Snehil Tiwari (1MS17IS153)

Institute: Ramaiah Institite of Technology, Bangalore

Date of submission: 07 May 2020
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

n_samples = 200 # No. of samples in the data

n_features = 5  # No. of features (dimensions of the data)

n_redundant = 1  # No. of redundent features

n_classes = 2  # No. of classes

# Synthesizing a Binary Classification Dataset Based on the Given requirements
X, y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_redundant=n_redundant, n_classes=n_classes, random_state=2)

# Creating a dataframe out of it
df = pd.DataFrame(X, columns=['feature1', 'feature2',
                              'feature3', 'feature4', 'feature5'])
df['label'] = y

g = sns.pairplot(df, hue="label")
print("Close the plot to continue!")
plt.show()

df.to_csv('./data/synthetic.csv', index=False)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=0)  # random state to ensure same results in every run

architecture = [
    # Input layer, input_dim=5 for the 5 features we have in our X data
    {'input_dim': 5, 'output_dim': 7, 'activation': 'relu'},
    # Hidden layer, with 7 neurons, this will take the data to a 7 dimensional space, but even a 3D would work here
    {'input_dim': 7, 'output_dim': 7, 'activation': 'relu'},
    # Output layer, output_dim=1, which will represent 0 or 1 for the 2 classes we have.
    {'input_dim': 7, 'output_dim': 1, 'activation': 'sigmoid'}
]


def init_layers(architecture):
    no_of_layers = len(architecture)
    params = {}

    for index, layer in enumerate(architecture):
        layer_index = index + 1
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']

        params['W' + str(layer_index)] = np.random.randn(layer_output_size,
                                                         layer_input_size) * 0.1
        params['b' + str(layer_index)
               ] = np.random.randn(layer_output_size, 1) * 0.1

    return params


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(dA, x):
    """
    dA is the derivatives of the activation in the current layer
    """
    sig = sigmoid(x)
    return dA * sig * (1 - sig)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(dA, x):
    dZ = np.array(dA, copy=True)
    dZ[x <= 0] = 0
    return dZ


def single_layer_forward_propogation(A_prev, W_curr, b_curr, activation="relu"):

    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation == 'relu':
        activation_function = relu

    elif activation == 'sigmoid':
        activation_function = sigmoid

    A_curr = activation_function(Z_curr)

    return A_curr, Z_curr


def forward_propogation(X, params, architecture):

    memory = {}
    A_curr = X

    for index, layer in enumerate(architecture):
        layer_index = index + 1
        A_prev = A_curr

        current_activation_function = layer['activation']

        W_curr = params['W' + str(layer_index)]
        b_curr = params['b' + str(layer_index)]

        A_curr, Z_curr = single_layer_forward_propogation(
            A_prev, W_curr, b_curr, current_activation_function)

        memory['A' + str(index)] = A_prev
        memory['Z' + str(layer_index)] = Z_curr

    return A_curr, memory


def loss_function(y_hat, y):
    """
    This function implements a log loss between 2 distribution samples
    Log loss gives a measure of the difference between 2 probability distributions, as we are trying to estimate the probability distribution of y here,
    we get a measure of how far y_hat is from y.
    """
    m = y_hat.shape[1]
    loss = -(1 / m) * (np.dot(y, np.log(y_hat).T) +
                       np.dot(1-y, np.log(1-y_hat).T))
    return np.squeeze(loss)


def accuracy_function(y_hat, y):
    """
    We implement a simple function that checks how many classifications were made correctly, and calculate accuracy based on that
    """
    y_hat = np.round(y_hat)
    return (y_hat == y).all(axis=0).mean()


def single_layer_backpropogation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation='relu'):
    """
    dA_curr: derivatives of current layer's activation values
    w_curr: current layer's weights
    b_curr: current layer's biases
    Z_curr: current layer's calculations without activation applied
    A_prev: previous layer's activation values
    """

    m = A_prev.shape[1]

    if activation == 'relu':
        backward_activation_function = relu_derivative

    elif activation == 'sigmoid':
        backward_activation_function = sigmoid_derivative

    dZ_curr = backward_activation_function(dA_curr, Z_curr)

    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def backpropogation(y_hat, y, memory, params, architecture):
    """
    y_hat: predictions
    y: true values
    memory: Current forward propogation's activations and calculations
    params: Weights and biases of the network
    architecture: architecture of the network
    """

    gradients = {}
    y = y.reshape(y_hat.shape)

    dA_prev = -(np.divide(y, y_hat) - np.divide(1-y, 1-y_hat))

    for layer_index_prev, layer in reversed(list(enumerate(architecture))):
        layer_index_curr = layer_index_prev + 1

        activation_function_current = layer['activation']
        dA_curr = dA_prev

        A_prev = memory['A' + str(layer_index_prev)]
        Z_curr = memory['Z' + str(layer_index_curr)]
        W_curr = params['W' + str(layer_index_curr)]
        b_curr = params['b' + str(layer_index_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backpropogation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activation_function_current)

        gradients['dW' + str(layer_index_curr)] = dW_curr
        gradients['db' + str(layer_index_curr)] = db_curr

    return gradients


def update(params, gradients, architecture, lr):
    """
    This is the standard Gradient Descent step
    """
    for index, layer in enumerate(architecture):
        index += 1

        params['W' + str(index)] -= lr * gradients['dW' + str(index)]
        params['b' + str(index)] -= lr * gradients['db' + str(index)]

    return params


def valid(X, y, params, architecture):

    y_hat, _ = forward_propogation(X, params, architecture)
    loss = loss_function(y_hat, y)
    accuracy = accuracy_function(y_hat, y)

    return loss, accuracy


def train(X, y, architecture, epochs, lr, valid_data):
    params = init_layers(architecture)
    loss_history = []
    accuracy_history = []
    X_valid = valid_data[0].T
    y_valid = valid_data[1]

    print('\n------------ TRAINING ---------------\n')

    for i in range(epochs):
        y_hat, memory = forward_propogation(X, params, architecture)
        loss = loss_function(y_hat, y)
        loss_history.append(loss)
        accuracy = accuracy_function(y_hat, y)
        accuracy_history.append(accuracy)

        if i % 1000 == 0:
            print('---------- EPOCH %d -------------' % i)
            print('Training accuracy: %.3f' % accuracy)
            print('Training loss: %.3f' % loss)

            valid_loss, valid_acc = valid(
                X_valid, y_valid, params, architecture)

            print("\nValidation accuracy: %.3f" % valid_acc)
            print("Validation loss: %.3f\n\n" % valid_loss)

        gradients = backpropogation(y_hat, y, memory, params, architecture)
        params = update(params, gradients, architecture, lr)

    print("---------------- FINAL RESULTS ------------------")

    print('Final training accuracy: %.3f' % accuracy)
    print('Final training loss: %.3f' % loss)

    valid_loss, valid_acc = valid(X_valid, y_valid, params, architecture)

    print("\nFinal validation accuracy: %.3f" % valid_acc)
    print("Final validation loss: %.3f" % valid_loss)

    return params, loss_history, accuracy_history


learning_rate = 3e-2
epochs = 10000
params, loss_history, accuracy_history = train(X_train.T, y_train, architecture, epochs, learning_rate,
                                               valid_data=(X_valid, y_valid))

print('\n\n--------------- HYPERPARAMETERS AND LEARNT WEIGHTS --------------')
print('Learing rate  : ', learning_rate)
print('Epochs        : ', epochs)
print('Learnt parameters: \n', params)


def predict(X_sample, params=params, architecture=architecture):
    y_hat, memory = forward_propogation(X_sample, params, architecture)
    return np.round(y_hat)


X_sample = X_valid[1].reshape(1, -1).T
y_sample = y_valid[1]

y_pred = predict(X_sample)
print('\n\n---------- PREDICTING A DATAPOINT -----------')
print("Data point        : ", X_sample.reshape(-1))
print("True y value      : ", y_sample)
print("Predicted y value : ", y_pred.reshape(-1))


y_pred = predict(X_valid.T)
y_pred = y_pred.reshape(-1)

def report(y_true, y_pred):

    target_labels = ['class_0', 'class_1']

    classificationReport = classification_report(
        y_true, y_pred, target_names=target_labels)
    confusionMatrix = confusion_matrix(y_true, y_pred)
    oa = accuracy_score(y_true, y_pred)

    print("\n\n------------------------ CONFUSION MATRIX -------------------------")
    print(confusionMatrix)

    print("\n\n---------------------- CLASSIFICATION REPORT ----------------------")
    print(classificationReport)

    print('\n\n------------------------ OVERALL ACCURACY -------------------------')
    print(oa)

report(y_valid, y_pred)
