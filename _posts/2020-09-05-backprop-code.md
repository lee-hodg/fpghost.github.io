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

In this formulation, at each layer the weights matrices have dimensions 

$$
\text{dim}(\mathbf{W}^{[l]})=n_l \times n_{l-1}
$$

the bias matrices have dimensions (after broadcasting over training examples)

$$
\text{dim}(\mathbf{b}^{[l]})=n_l \times m
$$

and the activations also have this dimension

$$
\text{dim}(\mathbf{A}^{[l]})=n_l \times m
$$

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

We will implement a deep NN with $L$ layers. This function will take an array specifying how many nodes in each layer of our NN, e.g. `[5,4,3]` would mean the input layer has 5 nodes, the hidden layer has 4 nodes and the output layer has 3 nodes. It will randomly initialize the weights and biases matrices that we will need for such a NN and return them in a parameters dictionary.


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
        # Randomly init n_l x n_{l-1} matrix for weights
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        # Randomly init n_1 x 1 matrix for biases (will be broadcast to n_1 x m)
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
                
    return parameters
```

This results in the parameters dictionary looking something like

```python
{'W1': [[ 0.01788628,  0.0043651 ,  0.00096497, -0.01863493, -0.00277388],
        [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
        [-0.01313865,  0.00884622,  0.00881318,  0.01709573,  0.00050034],
        [-0.00404677, -0.0054536 , -0.01546477,  0.00982367, -0.01101068]],
 'W2': [[-0.01185047, -0.0020565 ,  0.01486148,  0.00236716],
        [-0.01023785, -0.00712993,  0.00625245, -0.00160513],
        [-0.00768836, -0.00230031,  0.00745056,  0.01976111]],
 'b1': [[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]],
 'b2': [[ 0.],
        [ 0.],
        [ 0.]]}
```

# Forward propagation

## Linear forward

First, implement the linear part. Namely

$$
\mathbf{Z}=\mathbf{W}\mathbf{A}+\mathbf{b}
$$

where here $\mathbf{A}$ is the activation from the previous layer (or if the first step, then the input data $\mathbf{X}$ as defined above).


```python

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
```

## Linear activation forward

The next forward step is applying activation to the nodes' output. We make this step somewhat generic by allowing the user to choose the activation as either sigmoid or ReLu


WHY IS THIS NOT TAKING Z FROM LINEAR FORWARD FUNCTION JUST DEFINED??

```python
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
```


## L layers

For this NN we want to have L-1 layers that are "Linear + ReLu" and the output layer to be "Linear + Sigmoid". The want the final layer to be sigmoid so that we can get predictions that represent probabilities for each of our classes with something like softmax.

<img src="/assets/images/model_architecture_kiank.png" alt="L-forward (stolen from Andrew Ng)" class="full">

