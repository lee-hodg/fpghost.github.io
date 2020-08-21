---
title: "Kelly Criterion"
date: 2020-08-20T20:00:00
categories:
  - blog
tags:
  - Trading
  - Betting
---



If we bet a fraction $f$ of our capital on each round, and a win resulted in a gain of $f\times b$, and a loss resulted in a loss of $f \times a$, then after $S$ winning rounds and $F$ losing rounds, $n=F+S$, we would have a portfolio worth

$$
W_n = (1+fb)^{S}(1-fa)^{F}
$$


We want to know what fraction $f$ of our capital would maximize that. Since the $log$ function is a monotonically increasing function (fancy speak for: it always increases as $x$ increases), it is equivalent to maximize $\log{W_n}$ and this will prove an easier task.


$$
\log{W_n} = S \log{(1+fb)} + F\log{(1-fa)}
$$

The thing we want to maximize is our expected wealth after $n$ rounds

$$
\mathbb{E}[\log{W_n}] = \mathbb{E}\left[S \log{(1+fb)} + F\log{(1-fa)}\right]
$$

On the right-hand side, by linearity of expectation of the fact that $\log{(1+fb)}$ and $\log{(1-fb)}$ are not random variables, just deterministic scalars, we can pull them out and simplify to:


$$
\mathbb{E}[\log{W_n}] = \mathbb{E}\left[S\right] \log{(1+fb)} + \mathbb{E}\left[F\right]\log{(1-fa)}
$$

If we have chance $p$ of a bet winning, and $(1-p)$ of it losing, then each bet is a Bernoulli trial and after $n$ rounds the results will be binomially distributed. The mean of the number of winning bets will therefore be $\mathbb{E}(S)=np$, and the mean of the losing bets will be $\mathbb{E}(F)=n(1-p)$


$$
\mathbb{E}[\log{W_n}] = np \log{(1+fb)} + n(1-p)\log{(1-fa)}
$$

## Jensen aside

For a strictly concave function like the log, [Jensen's Inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) tells us that

$$
g(\mathbb{E}(X)) > \mathbb{E}(g(X))
$$

so 

$$
\log{(\mathbb{E}(W_n))} > \mathbb{E}(\log{(W_n)})
$$

hence we are also maximizing the expected wealth $\mathbb{E}(W_n)$

## Maximising with the usual calculus way

Differentiating

$$
\frac{pb}{1+fb}-\frac{a(1-p)}{1-fa}=0
$$

Solving for $f$ with a bit of algebra gives

$$
f = \frac{pb-a(1-p)}{ab}
$$

This is the optimal fraction to bet given chances of winning and the payout scheme

## Examples

Coming soon...


