---
title: "COVID-19 Analysis: what can the data tell us?"
date: 2020-11-17T20:00:00
categories:
  - blog
tags:
  - Data-science
toc: true
toc_sticky: true
header:
  overlay_image: /assets/images/coronavirus.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  actions:
    - label: "Notebook"
      url: "https://github.com/lee-hodg/Covid19Study/blob/master/CovidStudy.ipynb"
---


This year, 2020, has been like no other in living memory. Coronavirus has shaken our collective sense of what normality means and will undoubtedly continue to affect the world for years to come in one way or another.

The purpose of this blog post is to try to gain some insights about Coronavirus using publically available datasets.

The data was obtained from [Our World In Data](https://ourworldindata.org/coronavirus), a scientific online publication whose research team is based at the University of Oxford and focuses on large global problems. 

# Loading the dataset

```python
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
```

Here we first import the libraries that we will use and next
we use the nice pandas `read_csv` method, which will load directly the data

```python
# import dataset and create a data frame
df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
```
# Data exploration

## At a glance

Taking a look at our data with `df.head()` shows

|    | iso_code   | continent   | location    | date       |   total_cases |   new_cases |   new_cases_smoothed |   total_deaths |   new_deaths |   new_deaths_smoothed |   total_cases_per_million |   new_cases_per_million |   new_cases_smoothed_per_million |   total_deaths_per_million |   new_deaths_per_million |   new_deaths_smoothed_per_million |   reproduction_rate |   icu_patients |   icu_patients_per_million |   hosp_patients |   hosp_patients_per_million |   weekly_icu_admissions |   weekly_icu_admissions_per_million |   weekly_hosp_admissions |   weekly_hosp_admissions_per_million |   total_tests |   new_tests |   total_tests_per_thousand |   new_tests_per_thousand |   new_tests_smoothed |   new_tests_smoothed_per_thousand |   tests_per_case |   positive_rate |   tests_units |   stringency_index |   population |   population_density |   median_age |   aged_65_older |   aged_70_older |   gdp_per_capita |   extreme_poverty |   cardiovasc_death_rate |   diabetes_prevalence |   female_smokers |   male_smokers |   handwashing_facilities |   hospital_beds_per_thousand |   life_expectancy |   human_development_index |
|---:|:-----------|:------------|:------------|:-----------|--------------:|------------:|---------------------:|---------------:|-------------:|----------------------:|--------------------------:|------------------------:|---------------------------------:|---------------------------:|-------------------------:|----------------------------------:|--------------------:|---------------:|---------------------------:|----------------:|----------------------------:|------------------------:|------------------------------------:|-------------------------:|-------------------------------------:|--------------:|------------:|---------------------------:|-------------------------:|---------------------:|----------------------------------:|-----------------:|----------------:|--------------:|-------------------:|-------------:|---------------------:|-------------:|----------------:|----------------:|-----------------:|------------------:|------------------------:|----------------------:|-----------------:|---------------:|-------------------------:|-----------------------------:|------------------:|--------------------------:|
|  0 | AFG        | Asia        | Afghanistan | 2019-12-31 |           nan |           0 |                  nan |            nan |            0 |                   nan |                       nan |                       0 |                              nan |                        nan |                        0 |                               nan |                 nan |            nan |                        nan |             nan |                         nan |                     nan |                                 nan |                      nan |                                  nan |           nan |         nan |                        nan |                      nan |                  nan |                               nan |              nan |             nan |           nan |                nan |  3.89283e+07 |               54.422 |         18.6 |           2.581 |           1.337 |          1803.99 |               nan |                 597.029 |                  9.59 |              nan |            nan |                   37.746 |                          0.5 |             64.83 |                     0.498 |
|  1 | AFG        | Asia        | Afghanistan | 2020-01-01 |           nan |           0 |                  nan |            nan |            0 |                   nan |                       nan |                       0 |                              nan |                        nan |                        0 |                               nan |                 nan |            nan |                        nan |             nan |                         nan |                     nan |                                 nan |                      nan |                                  nan |           nan |         nan |                        nan |                      nan |                  nan |                               nan |              nan |             nan |           nan |                  0 |  3.89283e+07 |               54.422 |         18.6 |           2.581 |           1.337 |          1803.99 |               nan |                 597.029 |                  9.59 |              nan |            nan |                   37.746 |                          0.5 |             64.83 |                     0.498 |
|  2 | AFG        | Asia        | Afghanistan | 2020-01-02 |           nan |           0 |                  nan |            nan |            0 |                   nan |                       nan |                       0 |                              nan |                        nan |                        0 |                               nan |                 nan |            nan |                        nan |             nan |                         nan |                     nan |                                 nan |                      nan |                                  nan |           nan |         nan |                        nan |                      nan |                  nan |                               nan |              nan |             nan |           nan |                  0 |  3.89283e+07 |               54.422 |         18.6 |           2.581 |           1.337 |          1803.99 |               nan |                 597.029 |                  9.59 |              nan |            nan |                   37.746 |                          0.5 |             64.83 |                     0.498 |
|  3 | AFG        | Asia        | Afghanistan | 2020-01-03 |           nan |           0 |                  nan |            nan |            0 |                   nan |                       nan |                       0 |                              nan |                        nan |                        0 |                               nan |                 nan |            nan |                        nan |             nan |                         nan |                     nan |                                 nan |                      nan |                                  nan |           nan |         nan |                        nan |                      nan |                  nan |                               nan |              nan |             nan |           nan |                  0 |  3.89283e+07 |               54.422 |         18.6 |           2.581 |           1.337 |          1803.99 |               nan |                 597.029 |                  9.59 |              nan |            nan |                   37.746 |                          0.5 |             64.83 |                     0.498 |
|  4 | AFG        | Asia        | Afghanistan | 2020-01-04 |           nan |           0 |                  nan |            nan |            0 |                   nan |                       nan |                       0 |                              nan |                        nan |                        0 |                               nan |                 nan |            nan |                        nan |             nan |                         nan |                     nan |                                 nan |                      nan |                                  nan |           nan |         nan |                        nan |                      nan |                  nan |                               nan |              nan |             nan |           nan |                  0 |  3.89283e+07 |               54.422 |         18.6 |           2.581 |           1.337 |          1803.99 |               nan |                 597.029 |                  9.59 |              nan |            nan |                   37.746 |                          0.5 |             64.83 |                     0.498 |



We can use

```
print(f'The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns')
```
to tell us about how many rows and columns our dataframe has:
```
The DataFrame has 57394 rows and 50 columns
```

We see that the data consists of multiple rows per country for various dates, and each entry specifies a lot of information about the state of coronavirus in that country at that date, such as the total cases recorded so far, the total deaths so far and so on.

## What columns do we have exactly?

`', '.join([col for col in df.columns])` shows us that we have 

```
iso_code, continent, location, date, total_cases, new_cases, new_cases_smoothed, total_deaths, new_deaths, new_deaths_smoothed, total_cases_per_million, new_cases_per_million, new_cases_smoothed_per_million, total_deaths_per_million, new_deaths_per_million, new_deaths_smoothed_per_million, reproduction_rate, icu_patients, icu_patients_per_million, hosp_patients, hosp_patients_per_million, weekly_icu_admissions, weekly_icu_admissions_per_million, weekly_hosp_admissions, weekly_hosp_admissions_per_million, total_tests, new_tests, total_tests_per_thousand, new_tests_per_thousand, new_tests_smoothed, new_tests_smoothed_per_thousand, tests_per_case, positive_rate, tests_units, stringency_index, population, population_density, median_age, aged_65_older, aged_70_older, gdp_per_capita, extreme_poverty, cardiovasc_death_rate, diabetes_prevalence, female_smokers, male_smokers, handwashing_facilities, hospital_beds_per_thousand, life_expectancy, human_development_index```
```

## What are the types for these columns?

```
df.dtypes
```

shows us this 

```
iso_code                               object
continent                              object
location                               object
date                                   object
total_cases                           float64
new_cases                             float64
new_cases_smoothed                    float64
total_deaths                          float64
new_deaths                            float64
new_deaths_smoothed                   float64
total_cases_per_million               float64
new_cases_per_million                 float64
new_cases_smoothed_per_million        float64
total_deaths_per_million              float64
new_deaths_per_million                float64
new_deaths_smoothed_per_million       float64
reproduction_rate                     float64
icu_patients                          float64
icu_patients_per_million              float64
hosp_patients                         float64
hosp_patients_per_million             float64
weekly_icu_admissions                 float64
weekly_icu_admissions_per_million     float64
weekly_hosp_admissions                float64
weekly_hosp_admissions_per_million    float64
total_tests                           float64
new_tests                             float64
total_tests_per_thousand              float64
new_tests_per_thousand                float64
new_tests_smoothed                    float64
new_tests_smoothed_per_thousand       float64
tests_per_case                        float64
positive_rate                         float64
tests_units                            object
stringency_index                      float64
population                            float64
population_density                    float64
median_age                            float64
aged_65_older                         float64
aged_70_older                         float64
gdp_per_capita                        float64
extreme_poverty                       float64
cardiovasc_death_rate                 float64
diabetes_prevalence                   float64
female_smokers                        float64
male_smokers                          float64
handwashing_facilities                float64
hospital_beds_per_thousand            float64
life_expectancy                       float64
human_development_index               float64
```
## Cleanup

In particular we want to ensure the data column is actually a pandas date type and not just a string

```
df.date = pd.to_datetime(df.date)
```

Now if we run `df.dtypes` again we'd see `date datetime64[ns]` as we desire.

## Missing data?

We can check out which our columns typically has a lot of missing data by running the command

```
(df.isnull().sum()/df.shape[0]).sort_values(ascending=False) * 100
```

Which will compute the percentage of null values per column and sort the columns by those with the highest percentage of null values at the top:


|                                    |         0 |
|:-----------------------------------|----------:|
| weekly_icu_admissions_per_million  | 99.378    |
| weekly_icu_admissions              | 99.378    |
| weekly_hosp_admissions             | 98.8762   |
| weekly_hosp_admissions_per_million | 98.8762   |
| icu_patients                       | 92.1769   |
| icu_patients_per_million           | 92.1769   |
| hosp_patients_per_million          | 91.2796   |
| hosp_patients                      | 91.2796   |
| new_tests                          | 62.0396   |
| new_tests_per_thousand             | 62.0396   |
| total_tests                        | 61.6388   |
| total_tests_per_thousand           | 61.6388   |
| tests_per_case                     | 60.2711   |
| positive_rate                      | 59.5585   |
| handwashing_facilities             | 57.8771   |
| new_tests_smoothed                 | 57.1175   |
| new_tests_smoothed_per_thousand    | 57.1175   |
| tests_units                        | 55.4448   |
| extreme_poverty                    | 41.5078   |
| reproduction_rate                  | 34.3207   |
| male_smokers                       | 31.7768   |
| female_smokers                     | 30.883    |
| total_deaths_per_million           | 23.1697   |
| total_deaths                       | 22.6958   |
| hospital_beds_per_thousand         | 19.9638   |
| stringency_index                   | 16.6341   |
| human_development_index            | 14.1949   |
| aged_65_older                      | 12.4212   |
| gdp_per_capita                     | 12.2434   |
| aged_70_older                      | 11.5448   |
| cardiovasc_death_rate              | 11.1179   |
| median_age                         | 11.0813   |
| diabetes_prevalence                |  7.86319  |
| total_cases_per_million            |  6.83521  |
| total_cases                        |  6.33516  |
| population_density                 |  5.2671   |
| new_deaths_smoothed_per_million    |  3.14841  |
| new_cases_smoothed_per_million     |  3.14841  |
| new_cases_smoothed                 |  3.03516  |
| new_deaths_smoothed                |  3.03516  |
| life_expectancy                    |  1.8434   |
| new_cases_per_million              |  1.73015  |
| new_deaths_per_million             |  1.73015  |
| new_deaths                         |  1.61864  |
| new_cases                          |  1.61864  |
| continent                          |  1.12555  |
| population                         |  0.562777 |
| iso_code                           |  0.562777 |
| date                               |  0        |
| location                           |  0        |


This tells us that the dataset seems to be quite lacking when it comes to information about ICU and hospital patients in particular (both weekly admissions and the counts of them in general), with over 90% of entries have null values for these columns. Following that around 60% of our rows are missing data about testing.

If we were going to use those columns as features to some ML algorithm we would have to think about how to handle those empty values. Whether to simply drop those columns or to impute the missing values using the mean for example.

# What countries are most affected by COVID-19?

To kick-off this analysis, let's look at which countries around the world seem to have suffered the most with COVID-19.

## By total cases

The first metric we will use to answer that question is the total case numbers per country.

As a convenience we define today's date as

```
today = pd.Timestamp("today").strftime("%Y-%m-%d")
top_N = 10
```

and then get the `top_N` countries by `total_cases` column as measured at today's date:

```
top_cases = df.loc[df['date'] == today, ['location', 'total_cases']].sort_values(by='total_cases', ascending=False)[1:top_N+1].reset_index(drop=True)
top_cases
```

Here `df.loc` selects rows matching today's date and keeps just the `location` and `total_cases` column. We then sort by the `total_cases` with the most cases at the top, and keep just the `top_N=10` countries. Finally the index is reset so that we see `0, 1, 2..,9` for example instead of the original indices.

The result is 


|    | location       |   total_cases |
|---:|:---------------|--------------:|
|  0 | United States  |   1.12055e+07 |
|  1 | India          |   8.87429e+06 |
|  2 | Brazil         |   5.87646e+06 |
|  3 | France         |   1.99123e+06 |
|  4 | Russia         |   1.97101e+06 |
|  5 | United Kingdom |   1.39068e+06 |
|  6 | Argentina      |   1.31837e+06 |
|  7 | Italy          |   1.20588e+06 |
|  8 | Colombia       |   1.20522e+06 |
|  9 | Mexico         |   1.0094e+06  |


The US has the highest number of cases in the world with over 11 million on the 17th Nov 2020, followed by India with almost 9M, and then Brazil with almost 6M.

Let's next compute how many worldwide cases there are

```
world_cases = df[(df['date'] == today) &  (df['location']=='World')].iloc[0]['total_cases']
print(f'There are {world_cases:,} worldwide cases')
```

Which tells us that there are  55,154,651 worldwide cases. 

Let's make this even clearer with a plot:

```
# create the matplotlib figure instance
fig, ax = plt.subplots(figsize=(10, 6))
sns.set()

# plot with seaborn
ax = sns.barplot('location', y='total_cases', data=top_cases,
                 palette='flare')
ax.set_title(f'COVID-19 - Top countries in number of cases - {today}', fontsize=14)
ax.set_xlabel('Country')

ax.set_ylabel('Total Cases (tens of millions)')
plt.tight_layout()
plt.savefig('graph1.png')
plt.xticks(rotation=45)
```

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/cov_graph1.png)

## Total cases per million

The previous plot was interesting, but if a country has a much greater population than another country then all things being equal we would still expect to have higher case numbers in that country. For example, one might argue that even if the US had locked down super early and implemented the strictest measures possible then they're bound to have more cases than a tiny country like Aruba, merely because of the difference in population.

For that reason, let's look at the total cases per million of the population.

Also given that some countries (such as Aruba) have utterly tiny populations per capita case numbers may not be so meaningful for them, so we filter out any countries with a population smaller than 10M. 

```python
# top  countries - cases
top_cases_per_m = df.loc[(df['date'] == today) & (df['population'] > 10000000), ['location', 'population', 'total_cases_per_million']].sort_values(by='total_cases_per_million', ascending=False)[1:top_N+1].reset_index(drop=True)
pd.set_option('float_format', '{:,}'.format)
top_cases_per_m
```

|    | location       |   population |   total_cases_per_million |
|---:|:---------------|-------------:|--------------------------:|
|  0 | United States  |  3.31003e+08 |                   33853.2 |
|  1 | France         |  6.52735e+07 |                   30506   |
|  2 | Argentina      |  4.51958e+07 |                   29170.2 |
|  3 | Peru           |  3.29718e+07 |                   28456.6 |
|  4 | Chile          |  1.91162e+07 |                   27861.4 |
|  5 | Brazil         |  2.12559e+08 |                   27646.2 |
|  6 | Netherlands    |  1.71349e+07 |                   26388.1 |
|  7 | Colombia       |  5.08829e+07 |                   23686.1 |
|  8 | Portugal       |  1.01967e+07 |                   22131.9 |
|  9 | United Kingdom |  6.7886e+07  |                   20485.5 |

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/cov_graph2.png)

## By deaths per million

Similarly, let's try to measure the impact of Covid-19 on a country by considering the death rate per million.

```python
top_deaths_per_m = df.loc[(df['date'] == today) & (df['population'] > 10000000), ['location', 'population', 'total_deaths_per_million']].sort_values(by='total_deaths_per_million', ascending=False)[1:top_N+1].reset_index(drop=True)
pd.set_option('float_format', '{:,}'.format)
top_deaths_per_m
```

|    | location       |   population |   total_deaths_per_million |
|---:|:---------------|-------------:|---------------------------:|
|  0 | Argentina      |  4.51958e+07 |                    790.494 |
|  1 | Brazil         |  2.12559e+08 |                    781.024 |
|  2 | Chile          |  1.91162e+07 |                    777.508 |
|  3 | United Kingdom |  6.7886e+07  |                    768.155 |
|  4 | Mexico         |  1.28933e+08 |                    766.764 |
|  5 | Bolivia        |  1.1673e+07  |                    758.072 |
|  6 | Italy          |  6.04618e+07 |                    756.395 |
|  7 | United States  |  3.31003e+08 |                    746.882 |
|  8 | Ecuador        |  1.76431e+07 |                    737.741 |
|  9 | France         |  6.52735e+07 |                    690.234 |


![alt]({{ site.url }}{{ site.baseurl }}/assets/images/cov_graph3.png)

One other thing we can look at is the proportion of the worldwide deaths came from the top ten countries by death count:

```python
world_deaths = df[(df['date'] == today) &  (df['location']=='World')].iloc[0]['total_deaths']
world_deaths
top_deaths_sum = df.loc[(df['date'] == today), ['location', 'total_deaths']].sort_values(by='total_deaths', ascending=False)[1:top_N+1]['total_deaths'].sum()
top_deaths_sum
print(f'We have {world_deaths:,} worldwide deaths from of COVID on {today}. And {100*top_deaths_sum/world_deaths: .2f}% came from the {top_N} countries')
```

We have 1,328,537.0 worldwide deaths from of COVID on 2020-11-17. And  67.63% came from the 10 countries.


## Testing per thousand

Case numbers are bound to increase the more tests that are done by a country. Let's check out countries by how many tests per thousand of the population have been carried out.

Again we will filter out countries with populations less than 10M, and since the testing columns often have null data we need to be careful to select only rows with non-nulls for this column:

```python
df_tests = df[(df['total_tests_per_thousand'].isnull() == False) & (df['population'] > 10000000) ]
```

Each country will have reported the latest `total_tests_per_thousand` on different dates. For example the UK reported it last on 2020-11-12 and Russia on 2020-11-15, so we group by location and select the latest date we have for that location in the `df_tests` dataset:

```python
df_tests_per_k = df_tests.loc[df_tests.groupby('location').date.idxmax()][['location', 'date', 'total_tests_per_thousand']].sort_values(by='total_tests_per_thousand', ascending=False)[0:top_N].reset_index(drop=True)
df_tests_per_k
```

|    | location       | date                |   total_tests_per_thousand |
|---:|:---------------|:--------------------|---------------------------:|
|  0 | United States  | 2020-11-13 00:00:00 |                    501.625 |
|  1 | United Kingdom | 2020-11-12 00:00:00 |                    482.174 |
|  2 | Russia         | 2020-11-15 00:00:00 |                    473.582 |
|  3 | Belgium        | 2020-11-14 00:00:00 |                    473.328 |
|  4 | Portugal       | 2020-11-11 00:00:00 |                    379.613 |
|  5 | Australia      | 2020-11-15 00:00:00 |                    367.819 |
|  6 | Spain          | 2020-11-05 00:00:00 |                    321.321 |
|  7 | Italy          | 2020-11-15 00:00:00 |                    312.236 |
|  8 | Germany        | 2020-11-08 00:00:00 |                    298.511 |
|  9 | Canada         | 2020-11-15 00:00:00 |                    274.177 |

We see that the US is leading the way in tests performed per 1000, so it shouldn't really be that surprising that case numbers are higher too. This may partly explain why the US has the highest per million case count but not the highest per million death rate.

Let's try and further normalize the cases per million by the tests per thousand?

```python
df_tests_per_k = df_tests.loc[df_tests.groupby('location').date.idxmax()][['location', 'population', 'date', 'total_cases_per_million', 'total_tests_per_thousand']]

df_tests_per_k['total_cases_per_million_per_tests_per_thousand'] = df_tests_per_k['total_cases_per_million']/df_tests_per_k['total_tests_per_thousand']


df_tests_per_k = df_tests_per_k.sort_values(by='total_cases_per_million_per_tests_per_thousand', ascending=False)[0:top_N].reset_index(drop=True)
df_tests_per_k
```

|    | location           |   population | date                |   total_cases_per_million |   total_tests_per_thousand |   total_cases_per_million_per_tests_per_thousand |
|---:|:-------------------|-------------:|:--------------------|--------------------------:|---------------------------:|-------------------------------------------------:|
|  0 | Peru               |  3.29718e+07 | 2020-09-05 00:00:00 |                 20528.1   |                     19.123 |                                         1073.47  |
|  1 | Brazil             |  2.12559e+08 | 2020-09-19 00:00:00 |                 21147.9   |                     30.21  |                                          700.029 |
|  2 | Mexico             |  1.28933e+08 | 2020-11-09 00:00:00 |                  7506.43  |                     17.149 |                                          437.718 |
|  3 | Bolivia            |  1.1673e+07  | 2020-11-13 00:00:00 |                 12241     |                     29.393 |                                          416.458 |
|  4 | Ecuador            |  1.76431e+07 | 2020-11-13 00:00:00 |                 10061.4   |                     30.555 |                                          329.287 |
|  5 | Colombia           |  5.08829e+07 | 2020-11-11 00:00:00 |                 22732.1   |                     89.101 |                                          255.127 |
|  6 | Guatemala          |  1.79156e+07 | 2020-11-13 00:00:00 |                  6337.67  |                     26.627 |                                          238.017 |
|  7 | Dominican Republic |  1.08479e+07 | 2020-11-09 00:00:00 |                 12000.7   |                     58.011 |                                          206.869 |
|  8 | Madagascar         |  2.7691e+07  | 2020-11-07 00:00:00 |                   621.609 |                      3.322 |                                          187.119 |
|  9 | Bangladesh         |  1.64689e+08 | 2020-11-15 00:00:00 |                  2613.99  |                     15.417 |                                          169.552 |


When normalized in terms of how many tests performed Peru and Brazil come out at the top. The US is all the way down at number 34 and the UK at 45. These are countries where a lot of testing was done and case numbers are also high. Brazil, on the other hand, did relatively few tests, but also has high case counts per million.

This is not to say however that the impact on the US and UK is solely an artefact of more testing; as we have seen, these countries also show very high in the death rate - which is obviously independent of how many tests a country does.


# The increase in Covid 

## Evolution in terms of case numbers

We can get an idea of the speed covid spread per country by plotting the time series of the case count. To keep the charts uncluttered we will do this for just the 5 countries with the most cases

```python
top_evo_cases = df.copy()
top_evo_cases.set_index('location', inplace = True)
top_evo_cases = top_evo_cases.loc[top_5_cases_country_names]
top_evo_cases = top_evo_cases.reset_index()

# create the matplotlib figure instance
fig, ax = plt.subplots(figsize=(18, 9))

# plot with seaborn
ax = sns.lineplot(x='date', y='total_cases', hue='location', data=top_evo_cases,  palette='colorblind');
ax.set_title('COVID-19 - Total of cases per country', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Number of Cases (ten million)')

plt.tight_layout()
plt.savefig('cov_evo_cases.png')
```


![alt]({{ site.url }}{{ site.baseurl }}/assets/images/cov_evo_cases.png)

The number of cases starting increasing around mid to late March for the US, with upticks around July and again in October, showing no sign of levelling off.

For a long time, Brazil, was the country with the second-highest number of cases until it was overtaken by India in early September. 

The UK seemed to flatten out for a while but has recently seen the case numbers begin to rise again.

## Evolution in terms of death counts

We take the top countries by per million death counts and plot the evolution of total deaths in the country over time

```python
top_evo_death = df.copy()
top_deaths_country_names = top_deaths_per_m['location'].values


top_evo_death.set_index('location', inplace = True)
top_evo_death = top_evo_death.loc[top_deaths_country_names]
top_evo_death = top_evo_death.reset_index()

# create the matplotlib figure instance
fig, ax = plt.subplots(figsize=(18, 9))

# plot with seaborn
ax = sns.lineplot(x='date', y='total_deaths', hue='location', data=top_evo_death, palette='colorblind');
ax.set_title('COVID-19 - Total of deaths per country', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Number of deaths')

plt.tight_layout()
plt.savefig('graph_death_evo.png')
```

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/graph_death_evo.png)


# A closer look at Brazil

Brazil has been something of an exception to most countries in the world. President Bolsonero is not a fan of lockdowns and the policy in Brazil was much more relaxed than
most countries around the globe. Borders reopened around July and tourism has been allowed since that time.


```python
# Data for just Brazil
df_brazil = df.loc[df['location'] == 'Brazil'].copy()
df_brazil.set_index('date', inplace=True)
```

When did Brazil record its first case and first death?

```
first_case_br = df_brazil.loc[df_brazil['total_cases'] == 1, ['total_cases']].sort_values(by='date').head(1)
first_death_br = df_brazil.loc[df_brazil['total_deaths'] == 1, ['total_deaths']].sort_values(by='date').head(1)
```

Shows us that the first case was on 2020-02-26 and the first death was on 
2020-03-18 

## Let's check the lag between cases reported and total deaths for Brazil


```python

# create the matplotlib figure instance
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(17, 7))

# linear scale
ax[0].plot('total_deaths', data=df_brazil)
ax[0].plot('total_cases', data=df_brazil)
ax[0].set_title('Evolution of COVID-19 in Brazil - Linear Scale', fontsize=14)
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Number of Cases/Casualties (million)')
ax[0].legend()

# logarithmic scale
plt.yscale('log')
ax[1].plot('total_deaths', data=df_brazil)
ax[1].plot('total_cases', data=df_brazil)
ax[1].set_title('Evolution of COVID-19 in Brazil - Logarithmic Scale', fontsize=14)
ax[1].set_xlabel('Date')
ax[1].legend()

plt.tight_layout(pad=3.0)
plt.savefig('brazil_case_death_lag.png')
```

Since the case numbers are much more than the deaths, the second plot here uses a log scale on the y-axis to better highlight the relation in time between case numbers and deaths.

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/brazil_case_death_lag.png)

We see that there is a lag between total cases and total deaths that appears to be around 2 weeks, at least for the first few months of the pandemic.

# Economic impact

The data to measure the economic impact was obtained from OECD.Stat OECD's quarterly national accounts data, available at [OECD.stat] (https://stats.oecd.org/Index.aspx?DatasetCode=SNA_TABLE1#)

We will compute the percentage change in GDP compared with the same quarter of the previous year (Q2 2019) to try to get an idea of which countries suffered the most economic impact from Covid-19.

## Import the data

First we load the data with

```
df = pd.read_csv('./data/QNA_16112020232419872.csv')
```

This is a CSV file that you can generate at the OECD.stat link provided and then export.

|    | LOCATION   | Country   | SUBJECT   | Subject       | MEASURE   | Measure                                                               | FREQUENCY   | Frequency   | TIME    | Period   | Unit Code   | Unit   |   PowerCode Code | PowerCode   |   Reference Period Code |   Reference Period |       Value |   Flag Codes |   Flags |
|---:|:-----------|:----------|:----------|:--------------|:----------|:----------------------------------------------------------------------|:------------|:------------|:--------|:---------|:------------|:-------|-----------------:|:------------|------------------------:|-------------------:|------------:|-------------:|--------:|
|  0 | JPN        | Japan     | GFSPB     | Public sector | CARSA     | National currency, current prices, annual levels, seasonally adjusted | Q           | Quarterly   | 2017-Q4 | Q4-2017  | JPY         | Yen    |                6 | Millions    |                     nan |                nan | 2.74922e+07 |          nan |     nan |
|  1 | JPN        | Japan     | GFSPB     | Public sector | CARSA     | National currency, current prices, annual levels, seasonally adjusted | Q           | Quarterly   | 2018-Q1 | Q1-2018  | JPY         | Yen    |                6 | Millions    |                     nan |                nan | 2.77151e+07 |          nan |     nan |
|  2 | JPN        | Japan     | GFSPB     | Public sector | CARSA     | National currency, current prices, annual levels, seasonally adjusted | Q           | Quarterly   | 2018-Q2 | Q2-2018  | JPY         | Yen    |                6 | Millions    |                     nan |                nan | 2.86863e+07 |          nan |     nan |
|  3 | JPN        | Japan     | GFSPB     | Public sector | CARSA     | National currency, current prices, annual levels, seasonally adjusted | Q           | Quarterly   | 2018-Q3 | Q3-2018  | JPY         | Yen    |                6 | Millions    |                     nan |                nan | 2.80418e+07 |          nan |     nan |
|  4 | JPN        | Japan     | GFSPB     | Public sector | CARSA     | National currency, current prices, annual levels, seasonally adjusted | Q           | Quarterly   | 2018-Q4 | Q4-2018  | JPY         | Yen    |                6 | Millions    |                     nan |                nan | 2.78111e+07 |          nan |     nan |


Here the rows with subject "B1_GA" represent the GDP measured by the "output approach" to calculate GDP. It sums the gross value added of various sectors, plus taxes and less subsidies on products. This is the measure we will use to represent the GDP. Similarly "B1_GE" is the GDP computed by the expenditure approach and "B1_GI" is the GDP computed by the income approach. 

The measure "CQRSA" represents millions of national currency, current prices, quarterly levels, seasonally adjusted.

We filter the dataframe by this 'B1_GA' and 'CQR' measure and also select out just rows in Q2 of 2019 and 2020

```python
fdf = df[(df['Country'].str.contains('Euro') == False) & (df['SUBJECT'] == 'B1_GA')  & (df['Period'].isin(['Q2-2019', 'Q2-2020'])) &  (df['MEASURE'] == 'CQR')]
```

Next set a multi-index on country and period

```python
fdf = fdf.set_index(['Country', 'Period'])
```

Next, create a dataframe representing the percentage change per country from Q2 2019 to Q2 2020 of the CQR measured GDP by expenditure

```python
data = []
for country in fdf.index.levels[0]:
    # Compute the Q2 % diff
    diff_q2 = (100*(fdf.loc[country].loc['Q2-2020']['Value'] - fdf.loc[country].loc['Q2-2019']['Value'])/fdf.loc[country].loc['Q2-2019']['Value'])
    # Append the data
    data.append([country, diff_q2])
    
# Generate a new diff with this data and sort by the diff    
ec_df = pd.DataFrame(data, columns=['country', 'gdp_diff_q2']).sort_values(by='gdp_diff_q2', ascending=False)
```

Finally, plot the bar-chart to visualize the impact per country

```python
# create the matplotlib figure instance
fig, ax = plt.subplots(figsize=(18, 18))

# plot with seaborn
sns.set(font_scale = 1.8)

ax = sns.barplot(x='gdp_diff_q2', y='country', data=ec_df, 
                  palette='colorblind');
ax.set_title('COVID-19 - Econonomic impact per country', fontsize=14)
ax.set_xlabel('GDP Difference Q2 2020 compared to Q2 2019')
ax.set_ylabel('Country')
   
    
plt.tight_layout()
plt.savefig('graph_econ_impact.png')
```


![alt]({{ site.url }}{{ site.baseurl }}/assets/images/graph_econ_impact.png)

It seems from this measure at least that China and Turkey actually experienced growth. Ireland and Scandinavian countries suffered less severe impact than other European countries, whereas Spain, Italy and France were very strongly affected. Saudi Arabia also appears to have took a huge economic blow in this quarter. We could speculate that this might be to do with global oil demand falling during global lockdowns. However, we may want to consider other measures of econonomic impact to be sure.

# Sweden: a closer look

