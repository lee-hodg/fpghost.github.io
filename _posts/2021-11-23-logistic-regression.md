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

## Stats aside

However, the more mathemtically rigorous answer is that this function comes from the MLE (Maximum Likelihood Estimator). Recall that if we have a Bernoulli model, then we can write

$$
L(x_1, \dots, x_n; p) = p^{\sum_{i=1}^{n} x_i}(1-p)^{n-\sum_{i=1}^{n} x_i}
$$

Recall we want the argmax over $p$ of this function, but since the logarithm is monatonically increasing this amounts to the same as finding the max over the log-likelihood

$$
\mathcal{l} = \log{L(x_1, \dots, x_n; p)} = \left(\sum_{i=1}^{n} x_i\right) \log{p} + \left(n-\sum_{i=1}^{n} x_i\right)\log({1-p})
$$

In regular statistics class, we'd now find the maxima by taking deratives and setting equal to 0.
This would give us the MLE estimator of our paramter $p$, which not too surprisngly would turn out to be the average

$$
p^{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i
$$


For a single training example the log-likelihood would be 

$$
\mathcal{l}_1 =  x \log{p} + \left(1-x\right)\log({1-p})
$$

and we not that finding the maxima of this function is equivalent to finding the minima of the negative. With a small change of notation, we see the negative is exactly our usual logistic regression cost function.

## Back to ML

### Backprop

We will need to know how our cost-function varies as our network weights vary.
To that end let's compute a few things

$$
\frac{\partial\mathcal{L}}{\partial a} = -frac{y}{a}+\frac{(1-y)}{(1-a)}
$$