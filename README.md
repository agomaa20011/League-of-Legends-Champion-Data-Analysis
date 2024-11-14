# League-of-Legends-Champion-Data-Analysis

<p align="center">
    <img width="400" src="https://github.com/agomaa20011/League-of-Legends-Champion-Data-Analysis/blob/main/League%20Of%20Legends.png">
</p>

This repository contains an analysis of League of Legends (LoL) champion data using a dataset that includes various attributes of the champions such as win rate, ban rate, popularity, and role. The analysis is performed using Python and various data science libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## Project Overview

In this project, the following analyses are performed:

### 1.Data Cleaning & Preprocessing:
1. Removing columns with more than 10% missing values.
2. Filling missing data with mean, mode, or custom values based on column type.
3. Correcting date formats and percentage columns for analysis.
### 2.Data Analysis & Visualization:
1. Time Series Analysis of champion releases over the years.
2. Correlation Heatmap to understand the relationship between win rate, ban rate, and popularity.
3. Box Plots to compare win rates and ban rates by champion role.
4. Top 10 Most Popular Champions with their win rates visualized in a bar chart.
### 3.Predictive Modeling:
1. Predicting ban rate based on win rate and popularity using linear regression.

## DATASET

this dataset is sourced from kaggle datasets
[DATASET LINK](https://www.kaggle.com/datasets/delfinaoliva/league-of-legends-champspopularity-winrate-kda)

## Python visuals

![Champion releases over time](https://github.com/agomaa20011/League-of-Legends-Champion-Data-Analysis/blob/main/Visuals/Champions%20Releases%20Over%20Time.png)

![the relationship between: winrate, banrate, popularity](https://github.com/agomaa20011/League-of-Legends-Champion-Data-Analysis/blob/main/Visuals/The%20relatiosn%20between%20winrate%2C%20banrate%2C%20popularity.png)

![winrate, banrate by champion](https://github.com/agomaa20011/League-of-Legends-Champion-Data-Analysis/blob/main/Visuals/winrate%2C%20banrate%20by%20role.png)

![Top pciked champions](https://github.com/agomaa20011/League-of-Legends-Champion-Data-Analysis/blob/main/Visuals/meta.png)


## Interactive Dashboard using power bi

![dashboard](https://github.com/agomaa20011/League-of-Legends-Champion-Data-Analysis/blob/main/dashboard/GENERAL%20DASHBOARD.PNG)

![dashboard](https://github.com/agomaa20011/League-of-Legends-Champion-Data-Analysis/blob/main/dashboard/GENERAL%20DASHBOARD%202.PNG)

![dashboard](https://github.com/agomaa20011/League-of-Legends-Champion-Data-Analysis/blob/main/dashboard/GENERAL%20DASHBOARD%203.PNG)


![ban rate by role](https://github.com/agomaa20011/League-of-Legends-Champion-Data-Analysis/blob/main/Visuals/ban%20rate%20by%20role%20prediction.png)

![actual vs. predicted ban rate](https://github.com/agomaa20011/League-of-Legends-Champion-Data-Analysis/blob/main/Visuals/actual%20vs%20predicted%20ban%20rate.png)
