from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero  (3073,10)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]   #500
    loss = 0.0
    for i in range(num_train): #500
        scores = X[i].dot(W)  #（1,10）
        # print("#######",y[i])
        correct_class_score = scores[y[i]] #y[i]为正确分类
        for j in range(num_classes): #10
            if j == y[i]:
                continue  #进入下一次循环
            margin = scores[j] - correct_class_score + 1  # note delta = 1,   w * X[i,j] - w * X[i,y[i]]+delta  j!=y[i]
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]  #X:(500,3073)
                # print(dW.shape,X[i].shape)
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    socres = np.matmul(X,W)
    # print(socres[range(0,N),y].shape)
    margin = socres - socres[range(0,N),y].reshape(-1,1) + 1
    margin[range(0,N),y] = 0
    margin = (margin > 0) * margin   #只保留大于0的，其余置0
    loss += np.sum(margin)/N        #loss : 1,1
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # W :(3073,10)  socres: (500,10) X (500,3073)

    counts = (margin > 0).astype(int)
    counts[range(N), y] = - np.sum(counts, axis=1)
    dW += np.dot(X.T, counts) / N + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
