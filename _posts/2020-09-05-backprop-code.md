---
title: "Vectorized Backprop: Coding it up"
date: 2020-09-05T16:06:00
categories:
  - Blog
tags:
  - ML
toc: true
toc_sticky: true
header:
  image: /assets/images/big_nn.svg
---

In a previous post, I walked through the maths of back-propagation ("backprop"). Here I will go through the implementation in Python (heavily based on Andrew Ng's course).

I'm going to use the alternative form equations (in the last blog post I denoted those with a tilde, but now I will drop that tilde). This means the design matrix expected is the transpose of the usual. Concretely it is the $k \times m$ dimensional matrix


$$
\mathbf{X}=\begin{pmatrix}
    \vert & \dots & \vert \\
    x^{(1)} & \dots  & x^{(m)}   \\
    \vert & \dots & \vert
\end{pmatrix}
$$

if we have $m$ training examples and $k$ features.

Foward-propagation looks like

$$
\mathbf{Z}=\mathbf{W}\mathbf{X}+\mathbf{b}
$$

or

$$
\begin{pmatrix}
    \vert & \dots & \vert \\
    z^{(1)} & \dots  & z^{(m)}   \\
    \vert & \dots & \vert
\end{pmatrix}=
\begin{pmatrix}
    \text{---} \hspace{-0.2cm} & \mathbf{w}^{[1]}_1 & \hspace{-0.2cm} \text{---} \\
    \vdots \hspace{-0.2cm} & \vdots & \hspace{-0.2cm} \vdots \\ 
    \text{---} \hspace{-0.2cm} & \mathbf{w}^{[1]}_{n_1} & \hspace{-0.2cm} \text{---}
\end{pmatrix}
\begin{pmatrix}
    \vert & \dots & \vert \\
    x^{(1)} & \dots  & x^{(m)}   \\
    \vert & \dots & \vert
\end{pmatrix}
+\quad
\begin{pmatrix}
    \vert & \dots & \vert \\
    b^{[1]} & \dots  & b^{[1]}   \\
    \vert & \dots & \vert
\end{pmatrix}
$$

where the bias is being broadcast.


This leads to the key equations:

{% capture notice-1 %}

$$
\begin{aligned}
\mathbf{dZ}^{[l]} &=\mathbf{dA}^{[l]}\circ g'\left(\mathbf{Z}^{[l]}\right)\quad \text{activation backwards}\\
\mathbf{dW}^{[l]}&
=\frac{1}{m} \left(\mathbf{dZ^T}\right)^{[l]} \mathbf{A}^{[l-1]} \quad \text{compute grad contrib}\\
\mathbf{db}^{[l]}&= \frac{1}{m}\sum_{i=1}^m \mathbf{dZ}^{[l](i)} \quad \text{compute bias grad contrib}\\
\mathbf{dA}^{[l-1]} &= (\mathbf{W}^T)^{[l]} \mathbf{dZ}^{[l]}  \quad \text{linear backwards}
\end{aligned}
$$
 
{% endcapture %}
<div class="notice">{{ notice-1 | markdownify }}</div>

# Helper functions


First we'll need some helper functions

Let's start with the activation functions. Recall that

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

and the rectified-linear unit (ReLu) is defined as



$$
f(x) =  \begin{cases}
   x, & \text{if } x>0\\
    0,    & \text{otherwise}
\end{cases}
$$

This implementation returns the original "cached" input as well for reasons of efficiency (we will need to cache those things in the feed-forward for use during the backprop)

```python
import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache
```

We also will need the derivatives to do the "activation backwards" steps.

Recall we had

$$
\mathbf{dZ}^{[l]} =\mathbf{dA}^{[l]}\circ g'\left(\mathbf{Z}^{[l]}\right)
$$

where this notation means element-wise multiplication.
For the sigmoid function its derivative is given by a nice formula

$$
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$

and for the ReLu activation the derivative is simply

$$
f'(x) =  \begin{cases}
   1, & \text{if } x>0\\
    0,    & \text{otherwise}
\end{cases}
$$


These will be used in the backwards step, and they will take `dA` passed from the subsequent layer, and also the cache of `Z`, which was stored during the feed-forward for reasons of efficiency as hopefully will become clearer later.


```python
def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache

    # This is dZ=dA*1
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    
    # dZ = dA * g'
    # dZ = dA * sigmoid(1-sigmoid)
    # Element-wise multiplication
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
```

# Initializing parameters

We will implement a deep NN with $L$ layers. This function will take an array specifying how many nodes in each layer of our NN, e.g. `[5,4,3]` would mean the input layer has 5 nodes, the hidden layer has 4 nodes and the output layer has 3 nodes.

```python

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    parameters = {} 
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters
```