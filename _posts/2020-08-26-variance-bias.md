---
title: "Sample variance"
date: 2020-08-26T14:03:00
categories:
  - blog
tags:
  - Statistics
---

# Where does the mysterious $(n-1)$ factor come from when computing the unbiased sample variance?

We have $n$ $X_i$ random variables that are identically distributed and independent from one another, with common mean $\mathbb{E}[X_i] = \mu_X$.


Let's say we constructed some estimator of the variance of these random variables:

$$
\tilde{S}_X = \frac{1}{n} \sum_{i}^n (X_i-\bar{X}_n)^2
$$

where

$$
\bar{X_n} = \frac{1}{n} \sum_{i}^n X_i
$$

is an estimator of the expectation.


To evaluate the expectated value of this estimator, first consider a term that will be useful to evaluate in advance:


$$
\begin{aligned}
&\mathbb{E}\left[\frac{1}{n}\left(\sum_{i}^n X_i\right)\left(\sum_{i}^n X_i\right)\right]\\
&=\frac{1}{n}\mathbb{E}\left[\sum_{i}^n X_i^2 + \sum_{i}^n\sum_{j\ne i}^n X_i X_j\right]\\
&=\frac{1}{n}\mathbb{E}\left[\sum_{i}^n X_i^2\right] + \frac{1}{n} \mathbb{E}\left[\sum_{i}^n\sum_{j\ne i}^n X_i X_j\right]\\
&=\frac{1}{n}\mathbb{E}\left[\sum_{i}^n X_i^2\right] + \frac{1}{n} \mathbb{E}\left[\sum_{i}^n X_i \sum_{j\ne i}^n X_j\right]
\end{aligned}
$$

Here all we did was seperated out the cross terms and then used linearity of expectations ($\mathbb{E}[aX+bY]=a\mathbb{E}[X]+b\mathbb{E}[Y]$)

Next, remembering that if $i\ne j$, then variables are all independent, so e.g. $\mathbb{E}[X_1 X_2]= \mathbb{E}[X_1]\mathbb{E}[X_2]$, but generally $\mathbb{E}[X_1^2] \ne (\mathbb{E}[X_1])^2$ since obviously $X_1$ is not independent from itself, and let's denote the common 2nd moment of the random variables as $\mu_{XX} = \mathbb{E}(X_{i}^2)$:

$$
\begin{aligned}
&=\frac{1}{n}\mathbb{E}\left[\sum_{i}^n X_i^2\right] + \frac{1}{n} \mathbb{E}\left[\sum_{i}^n X_i\right]\left[\sum_{j\ne i}^n X_j\right]\\
&=\frac{1}{n}\sum_{i}^n \mathbb{E}[X_i^2] + \frac{1}{n} \sum_{i}^n \mathbb{E}[X_i] \sum_{j\ne i}^n \mathbb{E}[X_j]\\
&=\frac{1}{n}n \mu_{XX} + \frac{1}{n} n \mu_X (n-1)\mu_X\\
&=\mu_{XX} + (n-1)\mu_X^2\\
\end{aligned}
$$

Now back to the original problem

$$
\begin{aligned}
\mathbb{E}[\tilde{S}_X] &= \frac{1}{n}  \mathbb{E}\left[\sum_{i}^n (X_i-\bar{X}_n)^2\right]\\
&=\frac{1}{n}\mathbb{E}\left[\sum_{i}^n X_i^2 \right]-\mathbb{E}\left[\bar{X}_n\sum_{i}^n \frac{X_i}{n}\right]-\mathbb{E}\left[\bar{X}_n\sum_{i}^n \frac{X_i}{n}\right]+\mathbb{E}\left[\bar{X}_n\bar{X}_n\right]\\
&=\frac{1}{n}\mathbb{E}\left[\sum_{i}^n X_i^2 \right]-\frac{1}{n}\mathbb{E}\left[\frac{\sum_{i}^n X_i\sum_{i}^n X_i}{n} \right]\\
&=\mu_{XX}-\frac{1}{n}\mathbb{E}\left[\frac{\sum_{i}^n X_i\sum_{i}^n X_i}{n} \right]\\
\end{aligned}
$$

Note $\bar{X}_n$ is the sample mean, itself a random variable. You cannot simply pull it out as $\mathbb{E}\left[\bar{X_n}\right] \ne \bar{X_n}$, unlike the population mean, which is a constant and for which $\mathbb{E}\left[\mu_X\right]=\mu_X$.

Now we can use the expression we evaluated earlier to get


$$
\begin{aligned}
\mathbb{E}[\tilde{S}_X] 
&=\mu_{XX}-\frac{1}{n}(\mu_{XX} + (n-1)\mu_X^2)\\
&=\frac{1}{n}\left[n\mu_{XX}-(\mu_{XX} + (n-1)\mu_X^2)\right]\\
&=\frac{n-1}{n}\left(\mu_{XX}-\mu_X^2\right)\\
&=\frac{n-1}{n}\left(\mathbb{E}(X^2)-\left(\mathbb{E}(X)\right)^2\right)\\
&=\frac{n-1}{n}\text{var}(X)
\end{aligned}
$$

We see finally that the estimator $\tilde{S}_X$ is a biased estimator of the population variance, but if we multiplied it by $n/n-1$ we'd get an unbiased estimator:

$$
\begin{aligned}
\hat{S}_X = \frac{1}{n-1}  \sum_{i}^n (X_i-\bar{X}_n)^2
\end{aligned}
$$

which has the property

$$
\mathbb{E}\left[\hat{S}_n\right] = \text{var}(X)
$$

as desired.