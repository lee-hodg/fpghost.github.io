---
title: "Logistic Regression from scratch"
date: 2021-11-23T20:00:00
categories:
  - blog
tags:
  - ML
  - Statistics
---

In simple Logistic Regression, we have the cost function

$$
\mathcal{L}(a, y) = -yln{(a)} - (1-y)ln{(1-a)}
$$

whereb $a$ is the predicted value and $y$ is the ground-truth label on the training set
(${0, 1}$).

Why this function? The intuitive version is that if $y=1$ the second term goes away, and the first
term that remains is small when $a\to 1$ (i.e. the predicted value approaches the correct ground truth), yet large when $a\to 0$ (i.e. we are punished with a large cost when the predicted answer goes away from the ground truth). Similarly for the other term.

However, the more mathemtically rigorous answer is that this function comes from the MLE (Maximum Likelihood Estimator). Recall that if we have a Bernoulli model, then we can write

$$
L(x_1, \dots, x_n; p) = p^{\sum_{i=1}^{n} x_i}(1-p)^{n-\sum_{i=1}^{n} x_i}
$$

Recall we want the argmax over $p$ of this function, but since the logarithm is monatonically increasing this amounts to the same as finding the max over the log-likelihood

$$
\mathcal{l} = \log{(L(x_1, \dots, x_n; p))} = \sum_{i=1}^{n} x_i \log{(p)} + (n-\sum_{i=1}^{n} x_i)\log({1-p})
$$