import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'

    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
        del X_train, y_train
        del X_test, y_test
        print('Clear previously loaded data.')
    except:
        pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False) #从数组中随机抽取元素
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # Normalize the data: subtract the mean image  ，归一化
    mean_image = np.mean(X_train, axis=0) #axis = 0：压缩行，对各列求均值，返回 1* n 矩阵  （1,3072）
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    # add bias dimension and transform into columns （ ，3073）
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])  #ones函数返回给定形状和数据类型的新数组，其中元素的值设置为1。此函数与numpy zeros（）函数非常相似
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('dev data shape: ', X_dev.shape)
print('dev labels shape: ', y_dev.shape)
# First implement the naive softmax loss function with nested loops.
# Open the file cs231n/classifiers/softmax.py and implement the
# softmax_loss_naive function.

# from classifiers.softmax import softmax_loss_naive
# import time
#
# # Generate a random softmax weight matrix and use it to compute the loss.
# W = np.random.randn(3073, 10) * 0.0001
# loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
#
# # As a rough sanity check, our loss should be something close to -log(0.1).
# print('loss: %f' % loss)
# print('sanity check: %f' % (-np.log(0.1)))
#
#
# # # Complete the implementation of softmax_loss_naive and implement a (naive)
# # # version of the gradient that uses nested loops.
# #
# # # As we did for the SVM, use numeric gradient checking as a debugging tool.
# # # The numeric gradient should be close to the analytic gradient.
# # from gradient_check import grad_check_sparse
# # f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
# # grad_numerical = grad_check_sparse(f, W, grad, 10)
# #
# # # similar to SVM case, do another gradient check with regularization
# # loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
# # f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
# # grad_numerical = grad_check_sparse(f, W, grad, 10)
#
#
#
# # Now that we have a naive implementation of the softmax loss function and its gradient,
# # implement a vectorized version in softmax_loss_vectorized.
# # The two versions should compute the same results, but the vectorized version should be
# # much faster.
# print('W  shape: ', W.shape)
# print('X  shape: ', X_dev.shape)
# print('y shape: ', y_dev.shape)
# tic = time.time()
# loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))
#
# from classifiers.softmax import softmax_loss_vectorized
# tic = time.time()
# loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))
#
# # As we did for the SVM, we use the Frobenius norm to compare the two versions
# # of the gradient.
# grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
# print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
# print('Gradient difference: %f' % grad_difference)

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.

from classifiers import Softmax

results = {}
best_val = -1
best_softmax = None

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained softmax classifer in best_softmax.                          #
################################################################################

# Provided as a reference. You may or may not want to change these hyperparameters
learning_rates = [1e-7, 2e-7,3e-7,4e-7,5e-7]
regularization_strengths = [2.5e4, 5e4]

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
# Load the raw CIFAR-10 data
cifar10_dir = 'datasets/cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
    del X_train, y_train
    del X_test, y_test
    print('Clear previously loaded data.')
except:
    pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
X_train_folds = []
y_train_folds = []
number = 1000
for i in range(5):
    X_train_folds.append(X_train[i * number : (i+1) * number])
    y_train_folds.append(y_train[i * number : (i+1) * number])
    X_train_folds[i] = np.reshape(X_train_folds[i], (X_train_folds[i].shape[0], -1))
    mean_image = np.mean(X_train_folds[i], axis=0)
    X_train_folds[i] -= mean_image
    X_train_folds[i] = np.hstack([X_train_folds[i], np.ones((X_train_folds[i].shape[0], 1))])

for lr in learning_rates:
    for reg in regularization_strengths:
        for i in range(5):
            X_tr = np.vstack(X_train_folds[0:i],X_train_folds[i+1:])
            y_tr = np.hstack(y_train_folds[0:i],y_train_folds[i+1:])
            X_te = X_train_folds[i]
            y_test = y_train_folds[i]
            W = np.random.randn(3073, 10) * 0.0001
            loss_vectorized, grad_vectorized = Softmax.softmax_loss_vectorized(W, X_tr, y_tr, 0.000005)



# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)