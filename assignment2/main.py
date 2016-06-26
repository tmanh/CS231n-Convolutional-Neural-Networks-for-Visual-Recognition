import time
import numpy as np
# import matplotlib.pyplot as plt

from cs231n.classifiers.fc_net import *
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.gradient_check import eval_numerical_gradient_array
from cs231n.solver import Solver


def TwoLayerNetDemo(reg=0.0):
    data = get_CIFAR10_data(9000, 1000)
    model = TwoLayerNet(reg=reg)
    solver = Solver(model, data, update_rule='sgd',
                    optim_config={'learning_rate': 1e-3, },
                    lr_decay=0.95, num_epochs=10,
                    batch_size=100, print_every=100)

    solver.train()

    X_test = data['X_test']
    y_test = data['y_test']
    num_samples = y_test.shape[0]

    acc = solver.predict(X_test, y_test, num_samples)
    print ["Accuracy", acc]


def FullyConnectedNetDemo(dropout=0.5, use_batchnorm=True, HeReLU=False,
                          weight_scale=1e-2, reg=0.0, update_rule='adam',
                          num_epochs=10):
    data = get_CIFAR10_data(19000, 1000)
    hidden_dims = [100, 50]
    model = FullyConnectedNet(hidden_dims=hidden_dims,
                              weight_scale=weight_scale,
                              use_batchnorm=use_batchnorm,
                              HeReLU=False, reg=reg)

    solver = Solver(model, data, update_rule=update_rule,
                    optim_config={'learning_rate': 1e-3, },
                    lr_decay=0.95, num_epochs=num_epochs,
                    batch_size=100, print_every=100)

    solver.train()

    X_test = data['X_test']
    y_test = data['y_test']
    num_samples = y_test.shape[0]

    acc = solver.predict(X_test, y_test, num_samples)
    print ["Accuracy", acc]


# just a "toy" demo
def ThreeLayerConvNetDemo(batch_size=32, num_filters=9, use_batchnorm=False,
                          weight_scale=1e-2, reg=0.0, update_rule='sgd'):
    data = get_CIFAR10_data(1000, 100)
    hidden_dims = [100, 50]
    model = ThreeLayerConvNet(num_filters=num_filters)

    solver = Solver(model, data, update_rule=update_rule,
                    optim_config={'learning_rate': 1e-3, },
                    lr_decay=0.95, num_epochs=10,
                    batch_size=batch_size, print_every=100)

    solver.train()

    X_test = data['X_test'][1:100]
    y_test = data['y_test'][1:100]
    num_samples = y_test.shape[0]

    acc = solver.predict(X_test, y_test, num_samples)
    print ["Accuracy", acc]


def main():
    print "Choose the demo:"
    print "1. Two Layers Networks"
    print "2. Fully Connected Networks"
    print "3. Convolutional Networks"

    try:
        str = input("Enter your input: ")
        print 'Received input is : %d' % str

        i = int(str)
        get_function(i)
    except ValueError as verr:
        print 'Please enter an integer'
    except Exception as ex:
        print 'Please enter an integer'


def wrong_value():
    print("Halt")


def get_function(argument):
    if argument == 1:
        TwoLayerNetDemo()
    elif argument == 2:
        FullyConnectedNetDemo()
    elif argument == 3:
        ThreeLayerConvNetDemo()
    else:
        wrong_value()

if __name__ == "__main__":
    main()
