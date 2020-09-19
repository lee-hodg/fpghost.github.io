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

# Asymptotics

If we have

$$
M(\theta) = (m_1(\theta), \dots, m_d(\theta))
$$

and after replacing expectations by sample averages we define

$$
\widehat{M}(\theta) = (\widehat{m_1}(\theta), \dots, \widehat{m_d}(\theta))
$$

Then let

$$
\Sigma(\theta) = \text{Cov}\left(X_1, X_1^2, \dots, X_1^d\right)
$$

be the covariance matrix of the random vector 

$$
\mathbf{X}=\left(X_1, X_1^2, \dots, X_1^d\right)
$$

Where does this come from?

Well, by the central-limit theorem (CLT) we have

$$
\sqrt{n}\left(\overline{X^k_n}-m_k(\theta)\right)\xrightarrow[n \to \infty]{(d))} \mathcal{N}(0, \text{var}\left(X_1^k\right))
$$

and more generally if I want to talk about the convergence of the entire random vector of moments, I have to talk about the full covariance matrix since by the multivariate CLT

$$
\sqrt{n}(\overline{X}_n - \mathbf{M}(\theta)) \xrightarrow[n \to \infty]{(d)} \mathcal{N}_d(\mathbf{0}, \Sigma_{\mathbf{X}})
$$

with

$$
\mathbf{\overline{X_n}}=\left(\overline{X_n}, \overline{X_n^2}, \dots, \overline{X_n^d}\right)
$$

These random variables are clearly correlates - if I know $X^2$ I know something about $X^4$ etc

This is nice, but the goal here is a CLT for the inverses, my MM estimators. The way to obtain this is by applying the delta method.

$$
\sqrt{n}(\widehat{\mathbf{M}} - \mathbf{M}(\theta)) \xrightarrow[n \to \infty]{(d)} \mathcal{N}_d(\mathbf{0}, \Sigma_{\mathbf{X}})
$$

we want to talk about the convergence of estimates parameters so

$$
\begin{aligned}
&\sqrt{n}\left[M^{-1}\left(\widehat{\mathbf{M}}\right) - M^{-1}\left(\mathbf{M}(\theta)\right)\right]\\
&=\sqrt{n}\left(\widehat{\mathbf{\theta}}^{MM} - \mathbf{\theta}\right)
\end{aligned}
$$

Recall that the multivariate delta method was

$$
\displaystyle  \displaystyle \sqrt{n} \left(\mathbf{g}(\bar{\mathbf{X}}_ n) - \mathbf{g}(\mathbf{\mu}) \right) \xrightarrow [n\to \infty ]{(d)} \, \displaystyle \mathcal{N}_d\left(\mathbf{0}, \nabla \mathbf{g}(\mathbf{\mu})^ T \Sigma _{\mathbf{X}} \nabla \mathbf{g}(\mathbf{\mu})\right)
$$

Here our mapping $g=M^{-1}$, so we will need to compute the gradient of that.

$$
\begin{aligned}
\sqrt{n}\left(\widehat{\mathbf{\theta}}^{MM} - \mathbf{\theta}\right)\xrightarrow [n\to \infty ]{(d)} \, \displaystyle \mathcal{N}_d\left(\mathbf{0}, \nabla \left(M^{-1}(\theta)\right)^ T \Sigma _{\mathbf{X}} \left(\nabla M^{-1}(\theta)\right)\right)
\end{aligned}
$$