---
title: "The Multivariate Delta Method"
date: 2020-08-29T14:51:00
categories:
  - blog
tags:
  - Statistics
toc: true
toc_sticky: true
---

Building on the univariate delta method 

## Gradient matrix of a vector function

Say we have some vector-valued function

$$
f: \mathbb{R}^d \to \mathbb{R}^k
$$

This is a function that takes a $d$ dimensional vector and spits out a $k$-dimensional vector. A special case could be when $k=1$ and we have a scalar-valued function of a vector.

Then the gradient matrix of this function $f$, denoted by $\nabla f$ is the $d\times k$ matrix


$$
\begin{aligned}
\nabla f&= \begin{pmatrix}
\vert & \vert & \vert & \vert\\
\nabla f_1 & \nabla f_2 & \dots & \nabla f_k\\
\vert & \vert & \vert & \vert
\end{pmatrix}\\
&=\begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_k}{\partial x_1} \\
\vdots & \dots & \vdots\\
\frac{\partial f_1}{\partial x_d} & \dots & \frac{\partial f_k}{\partial x_d} \\
\end{pmatrix}
\end{aligned}
$$

which is also the transpose of the Jacobian matrix $\mathbf{J}_f$

### Example

$$
f: \mathbb{R}^3 \to \mathbb{R}^2
$$

and 

$$
f(x, y, z) = \begin{pmatrix}
x+y \\
xy^2+z
\end{pmatrix}
$$

Then the gradient matrix would be

$$
\begin{aligned}
\nabla f &=\begin{pmatrix}
\frac{\partial f_x}{\partial x} & \frac{\partial f_y}{\partial x} \\
\frac{\partial f_x}{\partial y} & \frac{\partial f_y}{\partial y} \\
\frac{\partial f_x}{\partial z} & \frac{\partial f_y}{\partial z} \\
\end{pmatrix}\\
&=\begin{pmatrix}
1 & y^2 \\
1 & 2xy \\
0& 1 \\
\end{pmatrix}
\end{aligned}
$$


## Multivariate Delta Method

We have a sequence of random vectors $\mathbf{T}_1, \dots, \mathbf{T}_n$, which we can also denote as $(\mathbf{T}_n)_{n\ge 1}$, and this sequence satisfies

$$
\sqrt{n}(\mathbf{T}_n-\vec{\theta}) \xrightarrow[n \to \infty]{(\mathbb{d})} \mathbf{T}
$$

for some $\vec{\theta} \in \mathbb{R}^d$

Then if we have some function

$$
\mathbf{g}: \mathbb {R}^ d \to \mathbb {R}^ k
$$

which is continously differentiable at $\vec{\theta}$/ Then, for any vector $\mathbf{t}\in \mathbb{R}^d$, the first-order multivariate Taylor expansion at $\vec{\theta}$ gives

$$
\displaystyle  \mathbf{g}\left(\mathbf{t}\right) = \mathbf{g}(\vec{\theta }) + \nabla \mathbf{g}(\vec{\theta })^ T \left(\mathbf{t}- \vec{\theta }\right) + \left\|  \mathbf{t}- \vec{\theta } \right\| \, \mathbf{u}(\mathbf{t})
$$

where $\mathbf{u}(\mathbf{t})\to \mathbf{0}$ as $\mathbf{t}\to\vec{\theta}$

If now we replace $\mathbf{t}$ with a random vector $\mathbf{T}$. rearrange and multiply both sides by $\sqrt{n}$:

$$
\displaystyle  \displaystyle \sqrt{n}\left(\mathbf{g}\left(\mathbf{T}_ n\right) -\mathbf{g}(\vec{\theta }) \right)= \displaystyle  \nabla \mathbf{g}(\vec{\theta })^ T \left(\sqrt{n}\left(\mathbf{T}_ n - \vec{\theta }\right)\right) + \left\|  \sqrt{n}(\mathbf{T}_ n - \vec{\theta }) \right\| \, \mathbf{u}(\mathbf{T}_ n).
$$

### First term

Considering the convergence of each term on the right as $n \to \infty$, then firstly by definition

$$
\displaystyle  \displaystyle \displaystyle \sqrt{n} \left(\mathbf{T}_ n - \vec{\theta } \right) \xrightarrow [n\to \infty ]{(d)} \mathbf{T},
$$

which also implies

$$
\displaystyle  \displaystyle \left(\mathbf{T}_ n - \vec{\theta } \right) \xrightarrow [n\to \infty ]{(d)/(p)} \mathbf{0}.
$$

or

$$
\displaystyle  \displaystyle \mathbf{T}_n \xrightarrow [n\to \infty ]{(d)/(p)} \vec{\theta}
$$
(since convergence in distribution is stronger than in probability)

The first term, $\, \left(\nabla \mathbf{g}(\vec{\theta })\right)^ T \left(\sqrt{n}\left(\mathbf{T}_ n - \vec{\theta }\right)\right)\,$,  is a continuous function of $\left(\sqrt{n}\left(\mathbf{T}_ n - \vec{\theta }\right)\right)$, hence by the continous mapping theorem

$$
\displaystyle  \displaystyle \left(\nabla \mathbf{g}(\vec{\theta })\right)^ T \left(\sqrt{n}\left(\mathbf{T}_ n - \vec{\theta }\right)\right) \xrightarrow [n\to \infty ]{(d)}\left(\nabla \mathbf{g}(\vec{\theta })\right)^ T\,  \mathbf{T}
$$


### Second term


For the second term, the first factor $\left\|  \sqrt{n}\left(\mathbf{T}_ n - \vec{\theta }\right) \right\|$ is again a continuous function of $\sqrt{n}\left(\mathbf{T}_ n - \vec{\theta }\right)$ , and therefore


$$
\displaystyle  \displaystyle \left\|  \sqrt{n}\left(\mathbf{T}_ n - \vec{\theta }\right) \right\| \xrightarrow [n\to \infty ]{(d)}\left\|  \mathbf{T} \right\|  \qquad \text {by continuous mapping theorem}.
$$

The second factor in the second term is a continuous function of $\mathbf{T}_n$

$$
\displaystyle  \displaystyle \mathbf{u}\left(\mathbf{T}_ n\right)\xrightarrow [n\to \infty ]{(d)/(p)} \mathbf{u}(\vec{\theta })\, =\, \mathbf{0}\qquad \text {by continuous mapping theorem}.
$$

and by the fact that

$$
\displaystyle  \displaystyle \mathbf{T}_n \xrightarrow [n\to \infty ]{(d)/(p)} \vec{\theta}
$$

By (multivariate) Slutsky theorem, the entire second term converges to $\mathbf{0}$

$$
\displaystyle  \left\|  \sqrt{n}\left(\mathbf{T}_ n - \vec{\theta }\right) \right\| \, \mathbf{u}(\mathbf{T}_ n)\xrightarrow [n\to \infty ]{(d)/\mathbf{P}}\left\|  \mathbf{T} \right\| (\mathbf{0})\, =\, \mathbf{0}.
$$


### Combining

Finally, applying the (multivariate) Slutsky theorem to the sum of the two terms gives:

$$
\displaystyle  \nabla \mathbf{g}(\vec{\theta })^ T \left(\sqrt{n}\left(\mathbf{T}_ n - \vec{\theta }\right)\right) + \left\|  \sqrt{n}\left(\mathbf{T}_ n - \vec{\theta }\right) \right\| \, \mathbf{u}(\mathbf{T}_ n)\xrightarrow [n\to \infty ]{(d)}\nabla \mathbf{g}(\vec{\theta })^ T \mathbf{T}+ \mathbf{0}\, =\, \nabla \mathbf{g}(\vec{\theta })^ T \mathbf{T}.
$$

and we have

$$
\displaystyle  \displaystyle \sqrt{n}\left(\mathbf{g}\left(\mathbf{T}_ n\right) -\mathbf{g}(\vec{\theta }) \right)= \, \nabla \mathbf{g}(\vec{\theta })^ T \mathbf{T}.
$$


### Applying this to the sample average

If now $\mathbf{T}_n=\bar{\mathbf{X}}_n$, the sample average, and $\, \vec{\theta }=\mathbb E[\mathbf{X}].\, \,$, then the multivariate CLT gives $\mathbf{T}\sim \mathcal{N}_d(\mathbf{0}, \Sigma_{\mathbf{X}})$, and so in this case the delta method gives

$$
\begin{aligned}
\displaystyle  \displaystyle \sqrt{n} \left(\mathbf{g}(\bar{\mathbf{X}}_ n) - \mathbf{g}(\mathbf{\mu}) \right) & \xrightarrow [n\to \infty ]{(d)} \nabla \mathbf{g}(\mathbf{\mu})^T \mathcal{N}_d(\mathbf{0}, \Sigma_{\mathbf{X}})\,\\
& \sim \, \displaystyle \mathcal{N}_d\left(\mathbf{0}, \nabla \mathbf{g}(\mathbf{\mu})^ T \Sigma _{\mathbf{X}} \nabla \mathbf{g}(\mathbf{\mu})\right)
\end{aligned}
$$

where the last step follows from rules for affine transformations of the multidimensional Gaussian.