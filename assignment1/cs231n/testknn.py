import random
import time
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt


# start =time.time()
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Load the raw CIFAR-10 data.
cifar10_dir = 'datasets/cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
# print('Training data shape: ', X_train.shape)
# print('Training labels shape: ', y_train.shape)
# print('Test data shape: ', X_test.shape)
# print('Test labels shape: ', y_test.shape)


# # Visualize some examples from the dataset.
# # We show a few examples of training images from each class.
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):  #循环了10次
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)  #replace:True表示可以取相同数字，False表示不可以取相同数字,默认为True 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
#     for i, idx in enumerate(idxs):#7次
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()


# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))#0到4999
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]
print(1,y_test)

# Reshape the image data into rows
print(X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
print(X_train.shape)
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

from classifiers import KNearestNeighbor

# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
# classifier = KNearestNeighbor()
# classifier.train(X_train, y_train)

# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_two_loops.
#
# Test your implementation:
# dists = classifier.compute_distances_two_loops(X_test)
# print(dists.shape)


# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
# plt.imshow(dists, interpolation='none')
# plt.show()


# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
# y_test_pred = classifier.predict_labels(dists, k=5)
#
# # Compute and print the fraction of correctly predicted examples
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
#
# a = np.array([1,45])
# b = np.array([[2],[1]])
# # b = a[np.argpartition(a,5)[:5]]
# c = [1,2,3]
# print(b+a)
# # print(b)

# dists_one = classifier.compute_distances_one_loop(X_test)
# print(dists_one.shape)
# dists_two = classifier.compute_distances_no_loops(X_test)
#
# difference = np.linalg.norm(dists_one - dists_two, ord='fro')
# print('One loop difference was: %f' % (difference, ))
# if difference < 0.001:
#     print('Good! The distance matrice ices are different')
# end = time.time()
# print("time : ",end-start)


def time_function(f, *args):
   """
   Call a function f with args and return the time (in seconds) that it took to execute.
   """
   import time
   tic = time.time()
   dists = f(*args)
   toc = time.time()
   print(toc-tic)
   return dists


# two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
# print('Two loop version took %f seconds' % two_loop_time)
#
# one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
# print('One loop version took %f seconds' % one_loop_time)
#
# no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
# print('No loop version took %f seconds' % no_loop_time)

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
number = int(X_train.shape[0]/num_folds)
# len1 = 5000/num_folds
# for i in range(num_folds):
#    mask = list(i*10000+np.array(range(int(len))))
#    mask1 = list(i*1000+np.array(range(int(len1))))
#    X_train_folds.append(X_train[mask])
#    y_train_folds.append(y_train[mask])
#
for i in range(num_folds):
    X_train_folds.append(X_train[i * number : (i+1) * number])
    y_train_folds.append(y_train[i * number : (i+1) * number])
    X_train_folds[i] = np.reshape(X_train_folds[i], (X_train_folds[i].shape[0], -1))
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
classifier = KNearestNeighbor()
num_test = 10000
for k in k_choices:

   accuracies = []
   for i in range(num_folds):
      X_train = np.vstack(X_train_folds[0:i] + X_train_folds[i+1:])
      y_train = np.hstack(y_train_folds[0:i] + y_train_folds[i+1:])
      X_test = X_train_folds[i]
      y_test = y_train_folds[i]
      classifier.train(X_train,y_train)
      dists = classifier.compute_distances_no_loops(X_test)
      y_test_pred = classifier.predict_labels(dists,k)
      num_correct = np.sum(y_test_pred == y_test)
      accuracy = float(num_correct) / y_test.shape[0]
      accuracies.append(accuracy)

      # accuracy = float(np.sum(classifier.predict_labels(dists, k) == y_valid_cv)) / y_valid_cv.shape[0]
      # accuracies.append(accuracy)
   k_to_accuracies[k] = accuracies



# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
    print("############################")
# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k]*len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
print("end")

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=10)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))