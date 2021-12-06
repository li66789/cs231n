from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(X.shape[0]):
        f = X[i].dot(W)
        f -= np.max(f)
        loss = loss + np.log(np.sum(np.exp(f))) - f[y[i]]
        # print(np.log(np.sum(np.exp(f))) - f[y[i]])
        dW[:, y[i]] -= X[i]
        s = np.exp(f).sum()
        for j in range(W.shape[1]):
            dW[:, j] += np.exp(f[j]) / s * X[i]
    loss = loss/500 + 0.5 * reg * np.sum(W*W)
    dW = dW/500 + reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)  #[500,10]
    scores -= np.max(scores,axis=1).reshape(X.shape[0],1)
    s = np.exp(scores).sum(axis=1)
    loss = np.log(s).sum() - scores[range(500), y].sum()

    counts = np.exp(scores) / s.reshape(500, 1)
    counts[range(500), y] -= 1
    dW = np.dot(X.T, counts)

    loss = loss / 500 + 0.5 * reg * np.sum(W * W)
    dW = dW / 500 + reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
