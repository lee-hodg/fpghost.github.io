---
title: "Method of Moments"
date: 2020-09-19T13:03:00
categories:
  - Blog
tags:
  - Statistics
toc: true
toc_sticky: true
header:
  image: /assets/images/moments.png
---


# Definition

In the discrete case the $k$'th *population moment* is

$$
m_ k(\theta ):= \mathbb {E}[X^ k] = \sum _{x \in E} x^ k p_\theta (x)
$$

where $X_1, \ldots , X_ n \stackrel{iid}{\sim } \mathbf{P}_{\theta ^*}$ and we have some statistical model $(E, \{ \mathbf{P}_{\theta }\} _{\theta \in \Theta })$

We don't know those population moments, but using the law of large numbers (LLN) a consistent (will tend to $m_k$ as the sample set gets large) estimator for $m_k(\theta)$ is

$$
\widehat{m}_ K(\theta ) = \displaystyle \frac{1}{n} \sum _{i =1}^ n X_i^k
$$


The definition is very similar in the continuous case.


# Convergence

By the LLN, each of the moments will converge, and we talk about a vector of the moments convering:

$$
\left(\widehat{m}_ 1, \dots, \widehat{m}_d \right)\xrightarrow[n \to \infty]{\mathbb{P}/a.s} \left(m_ 1, \dots, m_d \right)
$$

# Moments estimator

Let's think about a Guassian $(\mathbb {R}, \{ N(\mu , \sigma ^2)\} _{\mu \in \mathbb {R}, \sigma > 0})$.

The population moments are defined as

$$
m_ k(\mu , \sigma ) = \mathbb {E}[X^ k]
$$

The first moment is simply the mean

$$
m_1(\mu, \sigma)=\mu
$$

and the second moment

$$
m_2(\mu, \sigma) = \mathbb {E}[X^2] = (\mathbb {E}[X])^2 + \left(\mathbb {E}[X^2] - (\mathbb {E}[X])^2 \right) = \mu ^2 + \sigma ^2.
$$

Therefore we have 2 equations for our 2 parameters. We want to solve those equations in terms of the parameters, then use the LLN hammer to replace the population moments with empirical moments. That's it.

In this case

$$
\begin{aligned}
\hat{\mu}^{MM} &= \bar{X_n}\\
\hat{\sigma^2}^{MM}&=\bar{X^2_n}-(\bar{X_n})^2
\end{aligned}
$$

It's a very simple method, but more formally we could write the recipe as follows.

We are mapping our parameters to some vector of moments

$$
\begin{aligned}
M: &\Theta \to \mathbb{R}^d\\
&\theta \mapsto M(\theta)=(m_1(\theta), \dots, m_d(\theta))
\end{aligned}
$$

We then assume $M$ is 1-1 and invert to get those parameters in terms of the moments

$$
\theta=M^{-1}(m_1(\theta), \dots, m_d(\theta))
$$

Finally we use the LLN and replace the expectations by sample averages to get our estimator

$$
\hat{\theta}^{MM}_n=M^{-1}(\widehat{m}_1, \dots, \widehat{m}_d)
$$