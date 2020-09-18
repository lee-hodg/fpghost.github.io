---
title: "Fisher Information"
date: 2020-09-198T12:46:00
categories:
  - Blog
tags:
  - Statistics
toc: true
toc_sticky: true
header:
  image: /assets/images/big_nn.svg
---

# Log-Likelihood

If $f_\theta(x)$ is the PDF (probability density function) of the distribution $\mathbb{P}(\theta)$, then for a single observation the log-likelihood is just

$$
l(\theta)=\log{L_1(\mathbf{X_1}, \theta)}=\log{f_\theta(x_1)}
$$

(Keep in mind that $l(\theta)$ is a random function and depends on $\mathbf{X_1}$.)

If our parameter is multi-dimensional, i.e. $\theta \in \mathbb{R}^d$, then we can compute the gradient of $l(\theta)$ 

$$
\nabla l(\theta) \in \mathbb{R}^d
$$

($l$ is a scalar function and we gradient is respect to $\theta$, so it has dimensions $d \times 1$)

This is now a *random vector*.

## Covariance

We have this random vector $\nabla l(\theta)$ and so we can talk about the covariance of that random vector as usual

$$
\text{Cov}\left(\nabla l(\theta)\right)=\mathbb{E}\left[\nabla l(\theta)\nabla l(\theta)^T\right]-\mathbb{E}\left[\nabla l(\theta)\right]\mathbb{E}\left[\nabla l(\theta)\right]^T
$$

This is actually the *Fisher Information*, which is denoted by $I(\theta)$.

 Even though this definition was introduced out of nowhere, the aim of this post is to show how it is useful and it what contexts this quantity appears.

$\mathcal{I}(\theta )$ is a matrix of size $d \times d$.


### Theorem

The Fisher Information it turns out is also equal to minus the expectation of the [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix)(the matrix consisting of all second derivatives with respect to the parameters) of $l(\theta)$:

$$
\mathcal{I}(\theta )=\text{Cov}\left(\nabla l(\theta)\right)=-\mathbb{E}\left[\mathbb{H}l(\theta)\right]
$$

**Sanity check**: my log-likelihood is a concave function so the Hessian is a negative-definite matrix. Negative of this matrix is therefore positive. The covariance matrix is also positive-definite, so this makes sense. 


#### One dimensional case

$$
\theta \in \mathbb{R}
$$

Then we have simply the Fisher information:

$$
\mathcal{I}(\theta )=\text{Var}\left(l'(\theta)\right)=-\mathbb{E}\left[l''(\theta)\right]
$$


#### Deriving a useful formula for the Fisher Information (one dimension)

So our statistical model for some continuous distribution is $(\mathbb {R}, \{ \mathbf{P}_\theta \} _{\theta \in \mathbb {R}})$

and recall normalization of the PDF means

$$
\int _{-\infty }^\infty f_\theta (x) \,  dx = 1
$$


For the case of the single-observation log-likelihood

$$
\ell (\theta ) = \ln L_1(X, \theta ) = \ln f_\theta (X)
$$

We wish to compute

$$
\textsf{Var}(\ell '(\theta )) = \mathbb {E}[\ell '(\theta )^2] - \mathbb {E}[\ell '(\theta )]^2,
$$

##### Second term

We can differentiate to see

$$
\ell '(\theta ) = \frac{\partial }{\partial \theta } \ln f_\theta (X) = \frac{\frac{\partial }{\partial \theta } f_\theta (X)}{f_\theta (X)}.
$$

Therefore the expected value is

$$
\begin{aligned}
\mathbb {E}[\ell '(\theta )] &=\mathbb {E}\left[\frac{\frac{\partial }{\partial \theta } f_\theta (X)}{f_\theta (X)}\right] \\
&= \int _{-\infty }^\infty \left( \frac{\frac{\partial }{\partial \theta } f_\theta (x)}{f_\theta (x)} \right) f_\theta (x) \,  dx \\
&= \int _{-\infty }^\infty \frac{\partial }{\partial \theta } f_\theta (x) \,  dx \\
\end{aligned}
$$

Since we are allowed to interchange the integral and derivative (assuming "nice" functions)

$$
\int _{-\infty }^\infty \frac{\partial }{\partial \theta } f_\theta (x) \,  dx  =\frac{\partial }{\partial \theta } \int _{-\infty }^\infty f_\theta (x) \,  dx = \frac{\partial }{\partial \theta } 1 = 0.\tag{1}
$$

We can use the same trick for the 2nd derivative

$$
\int _{-\infty }^\infty \frac{\partial ^2 }{\partial \theta ^2} f_\theta (x) \,  dx = \frac{\partial ^2}{\partial \theta ^2} \int _{-\infty }^\infty f_\theta (x) \,  dx = \frac{\partial ^2}{\partial \theta ^2 } 1 = 0. \tag{2}
$$

This means that

$$
\mathbb {E}[\ell '(\theta )] = 0
$$

and we are reduced to 

$$
\mathcal{I}(\theta )= \textsf{Var}(\ell '(\theta )) = \mathbb {E}[\ell '(\theta )^2]
$$

##### What's left?

$$
\begin{aligned}
\mathcal{I}(\theta )=\mathbb {E}\left[(\ell '(\theta ))^2\right]
& = \mathbb {E}\left[\left( \frac{\frac{\partial }{\partial \theta }f_\theta (X)}{f_\theta (X)} \right)^2\right]\\
&= \int _{-\infty }^\infty \frac{(\frac{\partial }{\partial \theta } f_\theta (x))^2}{f^2_\theta (x)} f_\theta(x)\,  dx\\
&= \int _{-\infty }^\infty \frac{(\frac{\partial }{\partial \theta } f_\theta (x))^2}{f_\theta (x)} \,  dx. \tag{3}
\end{aligned}
$$


##### Alternate direction

The theorem we want to show is

$$
\mathcal{I}(\theta )=\text{Var}\left(l'(\theta)\right)=-\mathbb{E}\left[-l''(\theta)\right]
$$

so now let's consider 

$$
-\mathbb{E}\left[l''(\theta)\right]
$$

What is the second derivative of log-likelihood? We already had the first derivative as

$$
\ell '(\theta ) = \frac{\partial }{\partial \theta } \ln f_\theta (X) = \frac{\frac{\partial }{\partial \theta } f_\theta (X)}{f_\theta (X)}.
$$

so the second derivative by the chain and product rule is

$$
\begin{aligned}
\ell ''(\theta ) &=  \frac{\frac{\partial^2 }{\partial \theta^2 } f_\theta (X)}{f_\theta (X)} - \frac{\frac{\partial }{\partial \theta } f_\theta (X)}{f^2_\theta (X)}\times \frac{\partial }{\partial \theta } f_\theta (X)\\
 &=  \frac{\frac{\partial^2 }{\partial \theta^2 } f_\theta (X)}{f_\theta (X)} - \frac{\left(\frac{\partial }{\partial \theta } f_\theta (X)\right)^2}{f^2_\theta (X)}
\end{aligned}
$$

Back to its expectation

$$
\begin{aligned}
-\mathbb{E}\left[l''(\theta)\right]&=-\int _{-\infty }^\infty \left(\frac{\frac{\partial^2 }{\partial \theta^2 } f_\theta (X)}{f_\theta (X)} - \frac{\left(\frac{\partial }{\partial \theta } f_\theta (X)\right)^2}{f^2_\theta (X)}\right)f_\theta (X)\,  dx.\\
&=-\int _{-\infty }^\infty \frac{\partial^2 f_\theta (X)}{\partial \theta^2 }\, dx + \int _{-\infty }^\infty \frac{(\frac{\partial }{\partial \theta } f_\theta (x))^2}{f_\theta (x)} \,  dx.
\end{aligned}
$$

The first term is zero by (2) and we are left with

$$
-\mathbb{E}\left[l''(\theta)\right]= \int _{-\infty }^\infty \frac{(\frac{\partial }{\partial \theta } f_\theta (x))^2}{f_\theta (x)} \,  dx.
$$

but this is precisely the same as (3)

Hence in one dimensional we have established the theorem as desired

{% capture notice-1 %}

$$
\mathcal{I}(\theta )=\text{Var}\left(l'(\theta)\right)=-\mathbb{E}\left[l''(\theta)\right]
$$
 
{% endcapture %}
<div class="notice">{{ notice-1 | markdownify }}</div>

## Examples

So far we just defined this quantity $\mathcal{I}(\theta)$, which we called the Fisher Information and we provided in one dimension at least a theorem for it. We haven't yet said why the quantity is interesting.

