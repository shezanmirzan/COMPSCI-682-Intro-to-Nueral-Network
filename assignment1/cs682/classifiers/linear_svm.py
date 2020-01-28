import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #For the correct class, the impact on derivative from one image 
        #is negative of the image * (Number of times margin is > 1)
        dW[:, y[i]]+= -1 * X[i, :]
        #For every other class, the impact on derivative from one image 
        #is negative of the image if the margin of that class is greater
        #than 1
        dW[:, j]+=X[i, :]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW /= num_train
  #Add Regularization factor
  dW += 2 * reg*W     #Factor of 2 comes from differentiating W^(2) 

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  score_matrix = np.dot(X,W)
  num_train = X.shape[0]
  #Now we need to subtract the scores of correct class from every other class score
  #for each image
  num_train_range = np.linspace(0,num_train,num_train,endpoint=False,dtype = int)
  margins = np.maximum(0, score_matrix - np.expand_dims(score_matrix[num_train_range,y],axis=1) + 1 )
  #Set margins for correct class to zero for each image
  margins[num_train_range,y] = 0
  loss = np.sum(margins) / num_train

  loss += reg * np.sum(W * W) #Adding regularisation
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #Each pixel of an image X_i impacts the gradient by different multiplicative factor.
  #X_i impacts all incorrect classes with a multiplicative factor of 1 if margin for that
  #class is greater than 0. And X_i impacts the correct class with a multiplicative factor of 
  # -1 * (Total number of incorrect class with margin greater than 0). Rest would create zero impact.
  #mult_factor = np.empty(margins.shape)
  
  mult_factor = np.where(margins>0,1,0)
  mult_factor[num_train_range,y] = -1 * np.sum(mult_factor,axis = 1)
  
  dW = np.dot(np.transpose(X),mult_factor)
  dW /= num_train
  
  #Add Regularization factor
  dW += 2 * reg*W  #Factor of 2 comes from differentiating W^(2) 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
