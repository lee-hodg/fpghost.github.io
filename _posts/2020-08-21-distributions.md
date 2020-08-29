---
title: "Derivation of mean and standard deviation of common distributions"
categories:
  - Blog
tags:
  - math
  - statistics
toc: true
toc_sticky: true
---



# Bernoulli

Bernoulli random variables have probability $p$ of taking the value $1$ and probability $1-p$ of taking the value $0$, so the expected value is simply

$$
\mathbb{E}(X) = p \times 1 + (1-p) \times 0 = p
$$

In order to compute the variance, first compute the second moment

$$
\mathbb{E}(X^2) = p \times 1^2 + (1-p) \times 0^2 = p
$$

Now use the alternative form for variance:

$$
\text{var}(X) = \mathbb{E}(X^2) - (\mathbb{E}(X))^2 = p-p^2=p(1-p)
$$


# Binomial distribution

## Mean

Can consider a binomial random variable as just the sum of a bunch of independent Bernoulli random variables, $X_i \stackrel{iid}{\sim} \text{Ber}(p)$

$$
X=X_1+\dots+X_n
$$

This makes it easy to compute the mean using linearity of expectation for independent random variables


$$
\mathbb{E}(X)=\mathbb{E}(X_1+\dots+X_n)=\mathbb{E}(X_1)+\dots+\mathbb{E}(X_n)
$$

Using the Bernoulli result that $\mathbb{E}(X_i)=p$, this reduces to 

$$
\mathbb{E}(X)=np
$$

## Variance

One again by linearity of variance when the variables are independent

$$
\text{var}(X)=\text{var}(X_1+\dots+X_n)=\text{var}(X_1)+\dots+\text{var}(X_n)
$$

and then given they are all identically distributed as Bernoulli with common $p$, and $\text{var}(X_i) = p(1-p)$ then we find

$$
\text{var}(X)=np(1-p)
$$

# Poisson

## Mean

If we have a random variables distributed as $X \stackrel{iid}{\sim} \text{Poiss}(\lambda)$
 it has PMF

$$
p_X(k) = \frac{\lambda^k}{k!}e^{-\lambda}, \, k \in \mathbb{N}
$$

The expected value is 

$$
\mathbb{E}(X) = \sum_{k=0}^{\infty} k \frac{\lambda^k}{k!}e^{-\lambda}
$$

we can drop the zeroth term as it contributes nothing and also divide the numerator and denominator by the $k$

$$
\mathbb{E}(X) = \sum_{k=1}^{\infty}  \frac{\lambda^k}{(k-1)!}e^{-\lambda}
$$

Then factor out the terms constant in $k$:

$$
\mathbb{E}(X) = \lambda e^{-\lambda}\sum_{k=1}^{\infty}  \frac{\lambda^{k-1}}{(k-1)!}
$$

Now change variables $l=k-1$ (expand out to convince yourself this is true)

$$
\mathbb{E}(X) = \lambda e^{-\lambda}\sum_{l=0}^{\infty}  \frac{\lambda^l}{l!}
$$

But this sum is just the expansion of $e^{\lambda}$

$$
\mathbb{E}(X) = \lambda e^{-\lambda}e^{\lambda} = \lambda
$$

## Variance

Start with the 2nd moment

$$
\mathbb{E}(X^2) = \sum_{k=0}^{\infty} k^2 \frac{\lambda^k}{k!}e^{-\lambda}
$$

We can drop the zeroth term and pull out factors again

$$
\mathbb{E}(X^2) = \lambda e^{-\lambda} \sum_{k=1}^{\infty} k \frac{\lambda^{k-1}}{(k-1)!}
$$

Simple algebra $k=k-1 + 1$ to split the sum

$$
\mathbb{E}(X^2) = \lambda e^{-\lambda} \left( \sum_{k=1}^{\infty} (k-1) \frac{\lambda^{k-1}}{(k-1)!}+\sum_{k=1}^{\infty}  \frac{\lambda^{k-1}}{(k-1)!}\right)
$$

Drop the first term of the first sum since it's zero cancel the factors from numerator and denominator

$$
\mathbb{E}(X^2) = \lambda e^{-\lambda} \left( \lambda \sum_{k=2}^{\infty} \frac{\lambda^{k-2}}{(k-2)!}+\sum_{k=1}^{\infty}  \frac{\lambda^{k-1}}{(k-1)!}\right)
$$

Use change of variables $l=k-2$ and $m=k-1$

$$
\mathbb{E}(X^2) = \lambda e^{-\lambda} \left( \lambda \sum_{l=0}^{\infty} \frac{\lambda^{l}}{l!}+\sum_{m=0}^{\infty}  \frac{\lambda^{m}}{m!}\right)
$$

and note these sums are the expansion of $e^{\lambda}$ again

$$
\mathbb{E}(X^2) = \lambda e^{-\lambda} \left( \lambda e^{\lambda}+e^{\lambda}\right)=\lambda^2+\lambda
$$


The variance is

$$
\text{var}(X) = \mathbb{E}(X^2) - (\mathbb{E}(X))^2 = \lambda^2+  \lambda - \lambda^2 = \lambda 
$$

So the variance is also $\lambda$