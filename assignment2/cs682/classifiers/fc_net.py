from builtins import range
from builtins import object
import numpy as np

from cs682.layers import *
from cs682.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        output_relu1, cache_relu = affine_relu_forward(X,W1, b1 )
        scores, cache_softmax = affine_forward(output_relu1,W2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5*(self.reg*(np.sum(W1**2) + np.sum(W2**2)))
        loss = data_loss + reg_loss
        
        dX2, dW2, db2 = affine_backward(dscores, cache_softmax)
        dX1, dW1, db1 = affine_relu_backward(dX2, cache_relu)

        dW2 += self.reg*W2
        dW1 += self.reg*W1
        
        grads['W2'] = dW2
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['b2'] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        #Set parameters for the first hidden layer
        self.params['W1'] = weight_scale * np.random.randn(input_dim,hidden_dims[0])
        #Set bias parameters for all hidden layers
        for hidden_layer in range(len(hidden_dims)):
            self.params['b'+str(hidden_layer + 1)] = np.zeros(hidden_dims[hidden_layer])
        #Set bias parameter for last layer
        self.params['b'+str(self.num_layers)] = np.zeros(num_classes)
        #Set weight parameters for the subsequent hidden layer
        for hidden_layer in range(0,len(hidden_dims)-1):
            self.params['W'+str(hidden_layer + 2)] = weight_scale * np.random.randn(hidden_dims[hidden_layer], hidden_dims[hidden_layer+1])
        #Set weight parameters for the last layer
        self.params['W'+str(self.num_layers)] = weight_scale * np.random.randn(hidden_dims[-1],num_classes)
        #Print shape for debug
#         for key in self.params:
#             print(self.params[key].shape)
#             print(key)
            
        #Set batch Normalization parameters if use_batchnorm is True
        if (self.normalization=='batchnorm' or self.normalization=='layernorm' ):
            #Set for first layer
            self.params['gamma1'] = np.ones(input_dim)
            self.params['beta1'] = np.zeros(input_dim)
            
            #Set for hidden layers
            for hidden_layer in range(len(hidden_dims)):
                self.params['gamma'+str(hidden_layer+1) ] = np.ones(hidden_dims[hidden_layer])
                self.params['beta' +str(hidden_layer+1)] = np.zeros(hidden_dims[hidden_layer])
                
            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.ln_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        regSum = 0
        caches = []
        inputToLayer = [] #Stores the last layer output everytime
        for i in range(self.num_layers - 1):
            if(i==0):
                inputToLayer = X
            if(self.normalization=='batchnorm'):
                W = self.params['W%d' % (i + 1)]
                b =  self.params['b%d' % (i + 1)]
                gamma =  self.params['gamma%d' % (i + 1)]
                beta = self.params['beta%d' % (i + 1)]
                out1, fc_cache = affine_forward(inputToLayer, W, b)
                out2, bn_cache = batchnorm_forward(out1, gamma, beta, self.bn_params[i])
                out, relu_cache = relu_forward(out2)
                cache = (fc_cache, bn_cache, relu_cache)
            elif(self.normalization=='layernorm'):
                W = self.params['W%d' % (i + 1)]
                b =  self.params['b%d' % (i + 1)]
                gamma =  self.params['gamma%d' % (i + 1)]
                beta = self.params['beta%d' % (i + 1)]
                out1, fc_cache = affine_forward(inputToLayer, W, b)
                out2, ln_cache = layernorm_forward(out1, gamma, beta, self.ln_params[i])
                out, relu_cache = relu_forward(out2)
                cache = (fc_cache, ln_cache, relu_cache)
     
            else:
                W = self.params['W%d' % (i + 1)]
                b =  self.params['b%d' % (i + 1)]
                out, cache = affine_relu_forward(inputToLayer,W,b)
                #print(out)
                
            #Updating inputTolayer for next layer
            inputToLayer = out    
            
            #Appending value of cache for this layer
            caches.append(cache)
            
            #If dropout is applied
            if(self.use_dropout):
                out, cache = dropout_forward(inputToLayer, self.dropout_param)
                #print(out)
                #Updating inputTolayer for next layer
                inputToLayer = out
                #Appending value of cache for this layer
                caches.append(cache)
                
            regSum += np.sum(self.params['W%d' % (i + 1)] ** 2)
        i = i + 1;
        W = self.params['W%d' % (i + 1)]
        b =  self.params['b%d' % (i + 1)]
        scores,cache = affine_forward(inputToLayer, W,b)
        caches.append(cache)
        regSum += np.sum(W ** 2)
        regSum *= 0.5 * self.reg
        #print(regSum)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        loss += regSum
        lastCacheElement = -1
        
        #Backpropogating through the last layer
        dout, dW, db = affine_backward(dout, caches[lastCacheElement])
        grads['W'+str(self.num_layers)] = dW + self.reg*self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db
        lastCacheElement -= 1
        
        #Backpropogating through hidden layers
        for i in range(self.num_layers - 1, 0, -1):

            if(self.use_dropout):
                dout = dropout_backward(dout, caches[lastCacheElement])
                lastCacheElement -= 1
            
            if(self.normalization=='batchnorm'):
                fc_cache, bn_cache, relu_cache = caches[lastCacheElement]
                d1 = relu_backward(dout, relu_cache)
                d2, dgamma, dbeta = batchnorm_backward(d1, bn_cache)
                dout, dW, db = affine_backward(d2, fc_cache)
                grads['gamma'+str(i)] = dgamma
                grads['beta'+str(i)] = dbeta
                lastCacheElement -= 1
            elif(self.normalization=='layernorm'):
                fc_cache, ln_cache, relu_cache = caches[lastCacheElement]
                d1 = relu_backward(dout, relu_cache)
                d2, dgamma, dbeta = layernorm_backward(d1, ln_cache)
                dout, dW, db = affine_backward(d2, fc_cache)
                grads['gamma'+str(i)] = dgamma
                grads['beta'+str(i)] = dbeta
                lastCacheElement -= 1
            else:
                dout, dW, db = affine_relu_backward(dout, caches[lastCacheElement])
                lastCacheElement -= 1
            
            
                
            grads['W'+str(i)] = dW + self.reg*self.params['W'+str(i)]
            grads['b'+str(i)] = db
            
        
        
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
