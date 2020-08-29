---
title: "Multidimensional Central Limit Theorem"
date: 2020-08-29T20:11:48
categories:
  - blog
tags:
  - Statistics
header:
  image: /assets/images/multiv_gaussian.png
---


Instead of just having a single random variable $X$, we may have an experiment for which we are recording several random variables, which we can consider as a random *vector* in 

$$
\mathbf{X} = (X^{(1)}, \dots, X^{(d)})^T
$$

Here we have a $d$ dimensional vector.

{% capture notice-2 %}
Note that in the univariate case we had $X_1, X_2, \dots, X_n$, were the subscripts denoted the trial/sample in a repeat of the experiment (e.g. $i$th roll of a dice and a single number recording the result). In the multivariate case, we would get a vector of results *on every single trial*, e.g. 


$$
\begin{aligned}
\mathbf{X}_0 &= (X^{(1)}_0, \dots, X^{(d)}_0)^T\\
\vdots\\
\mathbf{X}_n &= (X^{(1)}_n, \dots, X^{(d)}_n)^T\\
\end{aligned}
$$
 
{% endcapture %}
<div class="notice">{{ notice-2 | markdownify }}</div>

## Gaussian Random Variables

Such a random vector is a Gaussian vector if any linear combination of its components is a univariate Gaussian variable.

In other words if $\alpha^T \mathbf{X}$ is a univariate Gaussian for *any* non-zero vector $\alpha \in \mathbb{R}^d$

This is much stronger than just demanding that the components individually are univariate Gaussians; we are demanding that any linear combination of them is Gaussian. Projecting the vector onto any $d$-dimensional vector $\alpha$ we still want a Gaussian.

The distribution of $\mathbf{X}$m the $d$-dimensional Gaussian distribution is completely specified by the mean $\mathbf{\mu} = \mathbb{E}[\mathbf{X}]$ (note that this is a vector whose components are the means of the components $(\mathbb{E}[X^{(1)}], \dots, \mathbb{E}[X^{(d)}])^T$) and also the $d\times d$ covariance matrix $\Sigma$, which captures how the components interact with one another. If $\Sigma$ is invertible, then the PDF of $\mathbf{X}$ is

$$
f_{\mathbf{X}}(\mathbf{x})=\frac{1}{\sqrt{(2\pi)^d\text{det}(\Sigma)}}\text{exp}\left[-\frac{1}{2}(\mathbf{x}-\mathbf{\mu})^T\Sigma^{-1}(\mathbf{x}-\mathbf{\mu})\right]
$$

Notice that the argument of the exponent has dimension $(1 \times d) \times (d \times d) \times (d \times 1) = 1 \times 1$, i.e. it is a scalar.

{% capture notice-3 %}

Note that if the components of our random vector are indeed independent, then the covariance matrix $\Sigma$ would be diagonal and the argument of the exponent would reduce to 

$$
\sum_{i=1}^d \frac{(\mathbf{x}_i-\mathbf{\mu}_i)^2}{2\sigma_i^2}
$$

and $\text{det}(\Sigma) = \prod_{i=1}^d \sigma_i^2$]

Meaning the PDF reduces to the product of the marginal PDFs, as expected for independent random variables


$$
f_{\mathbf{X}}(\mathbf{x})=\frac{1}{\sqrt{(2\pi)^d\sigma_1^2}}\text{exp}\left({-\frac{(\mathbf{x}_1-\mathbf{\mu}_1)^2}{2\sigma_1^2}}\right) \times \dots \times \frac{1}{\sqrt{(2\pi)^d\sigma_d^2}}\text{exp}\left({-\frac{(\mathbf{x}_d-\mathbf{\mu}_d)^2}{2\sigma_d^2}}\right)
$$

{% endcapture %}
<div class="notice">{{ notice-3 | markdownify }}</div>

## Affine transformations

