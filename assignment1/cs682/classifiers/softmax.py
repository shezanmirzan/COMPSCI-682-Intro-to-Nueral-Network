import numpy as np
from random import shuffle

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
  for i in range(num_train):
     
    temp = X[i,:].dot(W)
    #To ensure numerical stability we subtract the max of scores
    temp -= np.amax(temp)
    #Now the scores matrix is exp(f_{yi} - max f_{j}) / \sum{j} exp(f_{j} - max f_{j})
    scores = np.exp(temp)
    scores = scores / np.sum(scores)
    #Loss of a softmax is -log(score of correct label)
    loss -= np.log(scores[y[i]])
    
    #The correct class label impacts the gradient by (score - 1) and othes impact
    #the gradient by score of their corresponding label
    for j in range(W.shape[1]):
      if j == y[i]:     #Subtracting 1 from scores of correct label
        dW[:,j] += (scores[j]-1) * X[i,:]
      else:
        dW[:,j] += scores[j] * X[i,:]
        
  dW = dW / num_train
  loss = loss / num_train
  
  #Handling regularization factor in loss and gradient 
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W    #Factor of 2 comes from differentiating W^(2) 
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
  temp = np.dot(X,W)
  #To ensure numerical stability we subtract the max of scores
  temp -= np.expand_dims(np.amax(temp, axis = 1),axis = 1)
  #Now the scores matrix is exp(f_{yi} - max f_{j}) / \sum{j} exp(f_{j} - max f_{j})
  temp = np.exp(temp)
  scores = temp / np.expand_dims(np.sum(temp, axis = 1),axis=1)
  
  #Loss of a softmax is -log(score of correct label)
  num_train_range = np.linspace(0,num_train,num_train,endpoint=False,dtype = int)
  loss_matrix = -np.log(scores[num_train_range,y])
  loss = np.sum(loss_matrix)
  
  #The correct class label impacts the gradient by (score - 1) and othes impact
  #the gradient by score of their corresponding label
  scores[num_train_range,y] -= 1  #Subtracting 1 from scores of correct label
  dW = np.dot(np.transpose(X),scores)
                       
  dW = dW / num_train
  loss = loss / num_train
                       
  #Handling regularization factor in loss and gradient      
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W    #Factor of 2 comes from differentiating W^(2)                                  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

