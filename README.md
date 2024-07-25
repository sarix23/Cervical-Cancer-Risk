# Cervical Cancer Risk: A Case Study

**Authors:** Giulia Saresini (matr. 864967), Nicola Perani (matr. 864755), Sara Nava (matr. 870885)  
**University of Milano-Bicocca – Master's Degree Program in Data Science**

## Overview

Cervical cancer remains a significant global health issue, with approximately 11,000 new cases diagnosed annually in the U.S. This project utilizes a Kaggle dataset to develop a predictive model aimed at identifying high-risk cervical cancer cases that require a biopsy. The dataset contains 36 variables and undergoes comprehensive pre-processing to handle issues such as data imbalance. The study evaluates model robustness, variable selection, and confidence intervals, emphasizing the importance of balanced training data. Results reveal high accuracy, with the J48 algorithm demonstrating notable robustness and high Recall. Future work will focus on exploring ensemble methods and further fine-tuning models to enhance cervical cancer risk assessment.

## Table of Contents

1. [Introduction](#introduction)  
2. [Pre-processing](#pre-processing)  
   - [Categorical Variable Processing and Recoding](#categorical-variable-processing-and-recoding)  
   - [Missing Data Handling](#missing-data-handling)  
   - [Class Imbalance Problem](#class-imbalance-problem)  
   - [Dataset Partitioning](#dataset-partitioning)  
   - [Training Set Balancing](#training-set-balancing)  
   - [Data Scaling](#data-scaling)  
3. [Model Training](#model-training)  
   - [Classification Models](#classification-models)  
   - [Feature Selection and Data Scaling](#feature-selection-and-data-scaling)  
   - [Models Robustness Evaluation](#models-robustness-evaluation)  
   - [Confidence Interval Evaluation](#confidence-interval-evaluation)  
   - [Sensitivity and Specificity Evaluation](#sensitivity-and-specificity-evaluation)  
   - [ROC Curve Evaluation](#roc-curve-evaluation)  
4. [Test Results](#test-results)  
5. [Conclusions](#conclusions)  
6. [References](#references)

## Introduction

Cervical cancer, responsible for approximately 11,000 new cases of invasive cervical cancer diagnosed annually in the U.S., poses a significant threat to women's health globally. Despite a steady decrease in new cases over the past decades, this disease continues to claim the lives of about 4,000 women in the U.S. and over 300,000 women worldwide.

This project explores risk factors associated with cervical cancer to identify critical cases needing biopsy intervention. The dataset, sourced from Kaggle and the UCI Repository, provides comprehensive information across 36 variables, including age, sexual history, smoking habits, hormonal contraceptive use, and STD diagnoses.

## Pre-processing

### Categorical Variable Processing and Recoding

Categorical variables were converted from doubles to integers and then to strings to eliminate decimal places. Binary variables were encoded as 0 and 1, with special recoding for the `Biopsy` variable for clarity.

### Missing Data Handling

Attributes with over 30% missing data were excluded. Missing data for other attributes were imputed based on the median for numerical values and the mode for string variables.

### Class Imbalance Problem

The dataset exhibited a significant imbalance between positive (53 cases) and negative (700 cases) instances. This imbalance was addressed through various methods to ensure model performance and avoid false negatives.

### Dataset Partitioning

The dataset was partitioned using stratified sampling to maintain class distribution. 80% of the data was used for training and 20% for testing.

### Training Set Balancing

To address class imbalance in the training set, SMOTE was used for oversampling. The training set was evaluated both balanced and imbalanced to compare performance.

### Data Scaling

Data scaling was evaluated based on model requirements. Most models used original data, except for Support Vector Machines (SVM), which were standardized.

## Model Training

### Classification Models

Six classification models were trained: J48, Random Forest, Logistic Regression, Multilayer Perceptron (MLP), Naive Bayes, and Support Vector Machine (SVM).

### Feature Selection and Data Scaling

Feature selection was performed using Multivariate Filter Method. Data scaling was applied according to model needs, resulting in different variable sets for balanced and unbalanced data.

### Models Robustness Evaluation

Models were evaluated using 10-fold cross-validation to assess accuracy and robustness. Results showed high accuracy with variability across different models.

### Confidence Interval Evaluation

Confidence intervals for accuracy were assessed to evaluate model overfitting. Models generally showed close alignment between training and validation accuracies.

### Sensitivity and Specificity Evaluation

Sensitivity and specificity were evaluated to ensure effective prediction of the `Biopsy` variable. Models trained on balanced data generally showed higher sensitivity.

### ROC Curve Evaluation

ROC curves and AUC values were used to assess model performance. The Multilayer Perceptron (MLP) exhibited the highest AUC in balanced conditions, while Random Forest demonstrated robustness in imbalanced data.

## Test Results

### Models Trained on Balanced Data

| Model           | Accuracy | Recall |
|-----------------|----------|--------|
| J48             | 0.947    | 0.818  |
| Random Forest   | 0.940    | 0.545  |
| Logistic        | 0.954    | 0.818  |
| MLP             | 0.818    | 0.455  |
| Naive Bayes     | 0.940    | 0.818  |
| SVM             | 0.818    | 0.545  |

### Models Trained on Imbalanced Data

| Model           | Accuracy | Recall |
|-----------------|----------|--------|
| J48             | 0.940    | 0.818  |
| Random Forest   | 0.934    | 0.818  |
| Logistic        | 0.934    | 0.818  |
| MLP             | 0.818    | 0.636  |
| Naive Bayes     | 0.927    | 0.818  |
| SVM             | 0.927    | 0.636  |

## Conclusions

The project aimed to develop a predictive model for identifying high-risk cervical cancer cases. Despite satisfactory accuracy across models, the emphasis was placed on high Recall to effectively detect positive cases. The J48 model emerged as the most robust, demonstrating high Recall and overall performance.

Future work will explore ensemble methods and further fine-tuning of models to enhance cervical cancer risk assessment. Expanding the dataset and refining techniques will contribute to more effective predictive models and improved patient outcomes.

## References

1. [Kaggle Dataset](https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification)  
2. [Introduction to Balanced and Imbalanced Datasets](https://encord.com/blog/an-introduction-to-balanced-and-imbalanced-datasets-in-machine-learning/)  
3. [Unbalanced Data vs. Balanced Data](https://matloff.wordpress.com/2015/09/29/unbalanced-data-is-a-problem-no-balanced-data-is-worse/)  
4. [Imbalance Class Problem](https://stats.stackexchange.com/questions/227088/when-should-i-balance-classes-in-a-training-data-set)  
5. [SMOTE Technique](https://www.blog.trainindata.com/overcoming-class-imbalance-with-smote/)  
6. [Feature Scaling for SVM](https://forecast- egy.com/posts/does-svm-need-feature-scaling-or-normalization/)  
7. [Feature Scaling for Random Forest](https://forecastegy.com/posts/does-random-forest-need-feature-scaling-or-normalization/)  
8. [Models Requiring Normalization](https://www.yourdatateacher.com/2022/06/13/which-models-require-normalized-data/)  
9. [Ensemble Methods](https://corporatefinanceinstitute.com/resources/data-science/ensemble-methods/)  
10. [Fine Tuning](https://encord.com/blog/training-vs-fine-tuning/#h2)
