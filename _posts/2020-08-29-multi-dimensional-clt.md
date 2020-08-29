---
title: "Multidimensional Central Limit Theorem"
date: 2020-08-29T20:11:48
categories:
  - blog
tags:
  - Statistics
header:
  image: /assets/images/multiv_gaussian.jpg
---


Instead of just having a single random variable $X$, we may have an experiment for which we are recording several random variables, which we can consider as a random *vector* in 

$$
\mathbf{X} = (X^{(1)}, \dots, X^{(d)})^T
$$

Here we have a $d$ dimensional vector.

{% capture notice-2 %}
Note that in the univariate case we had $X_1, X_2, \dots, X_n$, were the subscripts denoted the trial/sample in a repeat of the experiment (e.g. $i$-th roll of a dice and a single number recording the result). In the multivariate case, we would get a vector of results *on every single trial*, e.g. 


$$
\begin{aligned}
\mathbf{X}_0 &= (X^{(1)}_0, \dots, X^{(d)}_0)^T\\
\vdots\\
\mathbf{X}_n &= (X^{(1)}_n, \dots, X^{(d)}_n)^T\\
\end{aligned}
$$
 
{% endcapture %}
<div class="notice">{{ notice-2 | markdownify }}</div>
