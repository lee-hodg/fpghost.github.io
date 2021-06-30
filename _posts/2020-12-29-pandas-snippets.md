---
title: "Useful Pandas Snippets"
date: 2020-12-29T07:19:00
categories:
  - Blog
tags:
  - Data science
header:
  image: /assets/images/imgproxy.jpg
  teaser: "/assets/images/imgproxy_teaser.jpeg"
---

# Imputing missing values 

## From the mean of a feature

| Country Name   | Year |  GDP |
|---|---|---|
| Aruba  | 1965  |  null |
|  Aruba |  1966 |  5.872478e+08 |

Say you have a dataframe for GDP by `Country Name` for each `year`, but some years are missing values. One way to deal with the missing values is to fill them in with the mean GDP for that country as follows:


```python
df['GDP_filled'] = df.groupby('Country Name')['GDP'].transform(lambda x: x.fillna(x.mean()))
```

## With forward fill

We can also use the [ffill](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.fillna.html) option from Pandas.

First we need to take care to sort the data by `year`, then we group by the `Country Name` so that the forward fill stays within each country

```python
df.sort_values('year').groupby('Country Name')['GDP'].fillna(method='ffill')
```

## With backward fill

Of course there is backward fill too:

```python
df.sort_values('year').groupby('Country Name')['GDP'].fillna(method='bfill')
```