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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    s_y_i_exp = np.exp(X[i].dot(W[:, y[i]]))
    sum_score_exp = 0
    for k in range(num_classes):
      sum_score_exp += np.exp(X[i].dot(W[:, k]))

    for k in range(num_classes):
      if k == y[i]:
        dW[:, k] += (s_y_i_exp/sum_score_exp-1)*X[i]
      else:
        dW[:, k] += (np.exp(X[i].dot(W[:, k]))/sum_score_exp)*X[i]
    loss += -np.log(s_y_i_exp/sum_score_exp)

  loss /= num_train
  loss += reg * np.sum(np.square(W))

  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]
  
  scores = X.dot(W)   # N by C matrix
  correct_scores = scores[np.arange(num_train), y]
  loss = np.sum(-np.log(np.exp(correct_scores)/np.sum(np.exp(scores), axis=1)))

  loss /= num_train
  loss += reg*np.sum(W*W)

  # construct prob matrix
  prob = np.exp(scores)/np.sum(np.exp(scores), axis=1).reshape(-1,1)
  prob[np.arange(num_train), y] -= 1

  dW = X.T.dot(prob)
  dW /= num_train
  dW += 2*reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

