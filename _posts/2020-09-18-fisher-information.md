---
title: "Fisher Information"
date: 2020-09-19T12:46:00
categories:
  - Blog
tags:
  - Statistics
toc: true
toc_sticky: true
header:
  image: /assets/images/fishing.jpg
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

The remaining terms are

$$
\begin{aligned}
\mathcal{I}(\theta )=\mathbb {E}\left[(\ell '(\theta ))^2\right]
& = \mathbb {E}\left[\left( \frac{\frac{\partial }{\partial \theta }f_\theta (X)}{f_\theta (X)} \right)^2\right]\\
&= \int _{-\infty }^\infty \frac{(\frac{\partial }{\partial \theta } f_\theta (x))^2}{f^2_\theta (x)} f_\theta(x)\,  dx\\
&= \int _{-\infty }^\infty \frac{(\frac{\partial }{\partial \theta } f_\theta (x))^2}{f_\theta (x)} \,  dx. \tag{3}
\end{aligned}
$$


##### Alternative direction

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

### Bernoulli

Let's say we have $X \sim \text{Ber}(p)$, then the likelihood is

$$
L(X, p) =  p^{x}(1-p)^{1-x}
$$

(this is just a trick to avoid writing the PMF of the discrete Bernoulli with braces - if we observe $x_1=1$ for example the element in the product would just reduce to $p$, and if we observe $x_1=0$, then the element in the product would reduce to $1-p$)

Therefore the log-likelihood is

$$
l(p)=x\log{p}+(1-x)\log{(1-p)}
$$

Taking derivatives with respect to the parameter $p$:

$$
l'(p)=\frac{x}{p}-\frac{1-x}{1-p}
$$

and the second derivative is

$$
l''(p)=-\frac{x}{p^2}-\frac{1-x}{(1-p)^2}
$$


### Via the variance

Recall

$$
\mathcal{I} = \text{var}(l'(p))
$$

so for Bernoulli

$$
\begin{aligned}
\mathcal{I}=\text{var}(l'(p))&=\text{var}\left(\frac{x}{p}-\frac{1-x}{1-p}\right)\\
&=\text{var}\left(x\left(\frac{1}{p}+\frac{1}{1-p}\right)-\frac{1}{1-p}\right)\\
&=\text{var}\left(x\left(\frac{1}{p}+\frac{1}{1-p}\right)\right)\\
&=\text{var}\left(\frac{x}{p(1-p)}\right)\\
&=\frac{1}{p^2(1-p)^2}\text{var}(X)\\
&=\frac{1}{p(1-p)}\\
\end{aligned}
$$

where the 3rd line follows from $\text{var}(X+c)=\text{var}(X)$ when $c$ is a constant, and the 4th line from $\text{var}(aX)=a^2\text{var}(X)$ where $a$ is also a constant. The final line follows because the variance of a Bernoulli random variable is $p(1-p)$

### Via the expectation

Recall

$$
\mathcal{I} = -\mathbb{E}(\ell''(p))
$$

so for the Bernoulli


$$
\begin{aligned}
\mathcal{I}&=-\mathbb{E}(\ell''(p))\\
&=-\mathbb{E}\left[-\frac{x}{p^2}-\frac{1-x}{(1-p)^2}\right]\\
&=\mathbb{E}\left[\frac{x}{p^2}+\frac{1-x}{(1-p)^2}\right]\\
&=\frac{p}{p^2}+\frac{1-p}{(1-p)^2}\\
&=\frac{1}{p}+\frac{1}{(1-p)}\\
&=\frac{1}{p(1-p)}
\end{aligned}
$$

Great. It matches.

### Binomial

Let's say we have $X \sim \text{Bin}(n, p)$, then the likelihood is

$$
L(X, p) =  {n \choose X} p^{X}(1-p)^{n-X}
$$

t in the product would reduce to $1-p$)

Therefore the log-likelihood is

$$
\displaystyle  \ell (p) \triangleq \ln {n \choose X} + X \ln p + (n-X) \ln (1-p), ~ ~ ~  X \in \{ 0,1,\dots ,n\} .
$$

Computing derivatives leads to

$$
\displaystyle  \ell '(p) = \frac{X}{p} - \frac{n-X}{1-p},
$$

and

$$
\displaystyle  \ell ^{\prime \prime }(p) = -\frac{X}{p^2} - \frac{n-X}{(1-p)^2}.
$$

We will compute the Fisher Information via the expectation

$$
\begin{aligned}
\displaystyle  \mathcal{I}(p)& = -\mathbb E[\ell ^{\prime \prime }(p)]\displaystyle = \mathbb E\left[\frac{X}{p^2} + \frac{n-X}{(1-p)^2}\right]\\
&=\displaystyle = \frac{np}{p^2} + \frac{n-np}{(1-p)^2}\\
&=\displaystyle = \frac{n}{p(1-p)}.
\end{aligned}
$$

where we used the fact that for a Binomial the expectation is $\mathbb{E}[X]=np$.


# Fisher Information and the Asymptotic Normality of the MLE

So, why is this an interesting quantity?

If we have some statistical model $(\mathbb {R}, \{ \mathbf{P}_\theta \} _{\theta \in \mathbb {R}})$, then the MLE (maximum likelihood estimator) for one observation maximizes the log-likelihood, which is the random variable $\ell (\theta ) = \ln L_1(X, \theta )$.

If we did the experiment and observed $X_1=x_1$, then consider the graph of $\theta \mapsto \ln L_1(x_1, \theta )$  (i.e. the $x_1$ is fixed and we graph over $\theta$), then the Fisher Information

$$
\mathcal{I}(\theta ) = -\mathbb E\left[\ell ^{\prime \prime }(\theta )\right]
$$

tells us how curved on average the log-likelihood is. The second-derivative measures concavity/convexity (how curved the function is at a particular point), and $\mathcal{I}(\theta)$ measures the *average* curvature of the $\ell(\theta)$.

It turns out that the Fisher information tells how curved (on average) the log-likelihood $\ln L_ n(x_1, \ldots , x_ n, \theta )$ for several samples $X_1 = x_1, \ldots , X_ n = x_ n$.

In particular, $\mathcal{I}(\theta ^*)$ tells how curved (on average) the log-likelihood is near the true parameter.

The larger the Fisher Information is near the true parameter, then the better the estimate we expect to have obtained from the MLE.

The asymptotic normality of the ML estimator, will turn out to depends upon the Fisher information. For a one-parameter model (like Bernoulli), the asymptotic normality result will say something along the lines of following: that the asymptotic variance of the ML estimator is inversely proportional to the value of Fisher information at the true parameter $\theta*$ of the statistical model. This means that if the value of Fisher information at $\theta*$ is high, then the asymptotic variance of the ML estimator for the statistical model will be low.

## Theorem

There is some broad set of conditions for which the ML estimator, no matter what the model - as long as it satisfies some conditions - will have asymptotic normality (just like a sample average has asymptotic normality using the CLT).

Remember that the ML estimator, does not *have to be the sample average*. Sometimes it works out that way, but it is not always that.
The following theorem for the converge of the MLE applies even in situations when the MLE is NOT the sample average.

We do need some conditions to hold however. If $\theta^* \in \Theta$ is the true parameter, then the conditions are

1. The parameter is *identifiable*.
2. For all $\theta \in \Theta$, the support of $\mathbb{P}_{\theta}$ does not depend on $\theta$ (think of the uniform distribution where the values could be $[0, a]$ and density is $1/a$. Clearly this violates this condition)
3. $\theta^*$ is not on the boundary of $\Theta$ (want to take derivatives and if on the boundary cannot do this)
4. $\mathcal{I}(\theta)$ is invertible in a neighbourhood of $\theta^*$
5. A few other "technical conditions".

Then $\widehat{\theta }_ n^{\text {MLE}}$ satisfies

$$
\widehat{\theta }_ n^{\text {MLE}} \xrightarrow[n \to \infty]{(\mathbb{P})}\theta* \quad \text{consistency}
$$

and

$$
\sqrt{n}\left(\widehat{\theta }_ n^{\text {MLE}} -\theta^*\right)\xrightarrow[n \to \infty]{(d)}\mathcal{N}_d\left(0, \mathcal{I}(\theta)^{-1}\right) \quad \text{asymptotic normality}
$$

From a semantic point of view, "information" says if I have a lot of information then the smaller the asymptotic variance. I reduce my uncertainty by having more information.


## Informal proof in one dimension

### Recap: the KL divergence and MLE

$$
\text{KL}(\mathbb{P}_{\theta^*}, \mathbb{P}_{\theta})=\mathbb{E}_{\theta^{*}}\left[\log{\frac{\mathbb{P}_{\theta^*}}{\mathbb{P}_{\theta}}}\right]
$$

and considering this as a function over $\theta$ that we'd like to minimize means we essentially have a constant and the term

$$
-\mathbb{E}_{\theta^{*}}\left[\log{\mathbb{P}_{\theta}}\right]
$$

By the LLN we replace the expectation by the sample average over our observations

$$
-\frac{1}{n}\sum_{i=1}^{n}\log{\left(\mathbb{P}_{\theta}(x_i)\right)}=-\log{\left(\prod_{i=1}^n\mathbb{P}_{\theta}(x_i)\right)}
$$

Minimizing the KL divergence over $\theta$ amounted to minimizing this negative log-likelihood.

In terms of the current notation, we have 

$$
\mathbb{E}[l(\theta)]
$$

and we are replacing that by

$$
\frac{1}{n}\sum_{i=1}^{n}\log{f_\theta(X_i)}
$$

Now

$$
l_i(\theta)=\log{f_\theta(x_i)}
$$


I know that

$$
\frac{\partial }{\partial \theta}\sum_{i=1}^n l_i(\theta) \bigg\rvert_{\theta=\theta^*}=\sum_{i=1}^n \ell'_i(\hat{\theta}) =0
$$

because $\hat{\theta}$ is a maximizer.

We also know that (see above)

$$
\mathbb {E}[\ell '(\theta^{*} )] = 0
$$

### Taylor expansion

We do a first-order Taylor expansion of the derivative

$$
\begin{aligned}
0 &= \sum_{i=1}^n \ell'_i(\hat{\theta})\\
&\approx\sum_{i=1}^n \ell'_i(\theta^{*})+(\hat{\theta}-\theta^{*})\ell_i''(\theta^{*})\\
&=\sum_{i=1}^n \ell'_i(\theta^{*})-\mathbb {E}[\ell '(\theta^{*} )]+(\widehat{\theta}-\theta^{*})\ell_i''(\theta^{*})
\end{aligned}
$$


where on the third line I've introduced the expectation, which is allowed since we know it is zero anyway.

This means I have a sum of random variables minus their expectations, so the CLT kicks in.

We are free to multiply by $1/\sqrt{n}$ - since the left-hand side is just 0


$$
\begin{aligned}
&\frac{1}{\sqrt{n}}\sum_{i=1}^n \left(\ell'_i(\theta^{*})-\mathbb {E}[\ell '(\theta^{*} )]\right)\\
&=\frac{1}{\sqrt{n}}n\left(\frac{1}{n}\sum_{i=1}^n \ell'_i(\theta^{*})\right)-\frac{n}{\sqrt{n}}\mathbb {E}[\ell '(\theta^{*} )]\\
&=\sqrt{n} \Bigg( \bar{\ell'}(\theta^{*})-\mathbb {E}[\ell '(\theta^{*} )]\Bigg)
\end{aligned}
$$

Then by the CLT

$$

\sqrt{n} \Bigg( \bar{\ell'}(\theta^{*})-\mathbb {E}[\ell '(\theta^{*} )]\Bigg)\xrightarrow[n \to \infty]{(d)}\mathcal{N}(0,\text{var}\left(\ell'(\theta^{*}\right))=\mathcal{N}\left(0,\mathcal{I}\left(\theta^{*}\right)\right)
$$

Putting this together

$$
\begin{aligned}
0&\approx \mathcal{N}\left(0,\mathcal{I}\left(\theta^{*}\right)\right)+(\widehat{\theta}-\theta^{*})\frac{1}{\sqrt{n}}\sum_{i=1}^n \ell_i''(\theta^{*})\\
&= \mathcal{N}\left(0,\mathcal{I}\left(\theta^{*}\right)\right)+\sqrt{n}(\widehat{\theta}-\theta^{*})\left(\frac{1}{n}\sum_{i=1}^n \ell_i''(\theta^{*})\right)\\
\end{aligned}
$$

and we know by LLN that

$$
\frac{1}{n}\sum_{i=1}^n \ell_i''(\theta^{*})\xrightarrow[n \to \infty]{(\mathbb{P})} \mathbb{E}\left[\ell''(\theta^{*})\right]=-\mathcal{I}(\theta^{*})
$$

so we can write

$$
\begin{aligned}
 \sqrt{n}(\widehat{\theta}-\theta^{*})&\xrightarrow[n \to \infty]{(d)}\frac{1}{\mathcal{I}(\theta^{*})}\mathcal{N}\left(0,\mathcal{I}\left(\theta^{*}\right)\right)\\
 &=\mathcal{N}\left(0,\frac{\mathcal{I}\left(\theta^{*}\right)}{\mathcal{I}^2\left(\theta^{*}\right)}\right)\\
  &=\mathcal{N}\left(0,\mathcal{I}^{-1}\left(\theta^{*}\right)\right)
 \end{aligned}
 $$

 where in the second line I've used $\text{var}(aX)=a^2 \text{var}(X)$.

 ## Applying this to Bernoulli

 For the case of Bernoulli, the ML estimator *is* just the sample average, since

 $$
\displaystyle  \ell_n '(p) = \frac{\sum_{i=1}^n x_i}{p} - \frac{n-\sum_{i=1}^n x_i}{1-p},
$$

and solving for $\ell_n '(\hat{p}^{MLE}_n) =0$ gives

$$
\hat{p}^{MLE}_n = \frac{1}{n}\sum_{i=1}^n x_i
$$

Given that the ML estimator is just the sample average this time, we already know from the CLT that

$$
 \sqrt{n}(\hat{p}^{MLE}_n-p^{*})\xrightarrow[n \to \infty]{(d)}\mathcal{N}(0, \sigma^2)
 $$

 where $\sigma^2=p^{*}(1-p^{*}$).

 However we also computed earlier that for a Bernoulli

$$
 \mathcal{I}(p)=\frac{1}{p(1-p)}
 $$

 This checks out with our more general equation for the asymptotic normality of an ML estimator in terms of inverse Fisher Information.
 
 Remember this more general form would apply even when the ML estimator wasn't just a sample average, and we can't just easily apply the CLT. But it's a nice sanity check that for the case where the ML estimator is just a sample average, we recover the expected result from the CLT.