# Earthquake-Damage-Prediction-ML
Machine Learning project for predicting earthquake building damage grades using EDA, preprocessing, classification models, and XGBoost.

# Earthquake Damage Prediction Using Machine Learning

## Project Overview

Earthquakes can cause severe damage to buildings depending on their structural characteristics, construction materials, location, age, and other factors.

This project focuses on predicting the **damage grade of buildings affected by an earthquake** using Machine Learning. The dataset contains information about buildings affected by the **Gorkha earthquake**, including building structure, foundation, roof type, geographic location, ownership, and other characteristics.

The objective is to build and compare multiple classification models and select the best-performing model for predicting the building damage grade.

---

## Problem Statement

The objective of this project is to predict the ordinal variable **`damage_grade`**, which represents the level of damage caused to a building during an earthquake.

There are three damage grades:

* **1** – Low damage
* **2** – Medium damage
* **3** – Almost complete destruction

The project also provides recommendations to seismologists and relevant authorities to help reduce significant building damage during future earthquake events.

---

# Tasks

### Task 1: Data Analysis

Perform complete data analysis on the earthquake building dataset, including:

* Understanding the dataset structure.
* Checking data types.
* Checking missing values.
* Checking duplicate records.
* Performing data preprocessing.
* Analyzing numerical and categorical features.
* Performing Exploratory Data Analysis (EDA).
* Identifying outliers.
* Studying feature relationships and correlations.
* Identifying important patterns related to building damage.

### Task 2: Predictive Model Development

Develop Machine Learning classification models to predict `damage_grade`.

The following models are evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

Hyperparameter tuning is performed using **GridSearchCV with 5-fold cross-validation** for the XGBoost model.

### Task 3: Suggestions to Seismologists

Provide recommendations based on the analysis and model results to reduce the risk of significant building damage during future earthquakes.

---

# Target Variable

### `damage_grade`

The target variable represents the level of earthquake damage suffered by a building.

| Damage Grade | Description                 |
| ------------ | --------------------------- |
| 1            | Low damage                  |
| 2            | Medium damage               |
| 3            | Almost complete destruction |

---

# Features

The dataset contains building-related and geographic features such as:

* Geographic region
* Number of floors
* Building age
* Building area
* Building height
* Land surface condition
* Foundation type
* Roof type
* Ground floor type
* Other floor type
* Building position
* Plan configuration
* Superstructure materials
* Legal ownership status
* Number of families
* Secondary building usage

These features are used to identify patterns associated with different earthquake damage grades.

---

# Project Workflow

1. Import Required Libraries
2. Load the Dataset
3. Merge Training Values and Labels
4. Understand the Dataset
5. Check Data Types
6. Check Missing Values
7. Check Duplicate Records
8. Perform Exploratory Data Analysis
9. Analyze Damage Grade Distribution
10. Analyze Numerical Features
11. Detect Outliers
12. Perform Correlation Analysis
13. Encode Categorical Variables
14. Separate Features and Target
15. Split Data into Training and Testing Sets
16. Perform Feature Scaling
17. Train Logistic Regression
18. Train Decision Tree
19. Train Random Forest
20. Train Gradient Boosting
21. Train XGBoost
22. Evaluate Model Performance
23. Compare Machine Learning Models
24. Perform XGBoost Hyperparameter Tuning
25. Use GridSearchCV with 5-Fold Cross-Validation
26. Select the Best Model
27. Generate Final Predictions
28. Provide Suggestions to Seismologists
29. Discuss Challenges Faced
30. Conclusion

---

# Data Preprocessing

The dataset was checked for:

* Missing values
* Duplicate records
* Incorrect data types
* Numerical and categorical features

Categorical variables were converted into numerical form using **One-Hot Encoding** with `drop_first=True`.

Feature scaling was performed using **StandardScaler**, particularly for Logistic Regression.

---

# Exploratory Data Analysis

The following visualizations and analyses were performed:

### Damage Grade Distribution

The distribution of the three damage grades was analyzed to understand the target variable.

### Numerical Feature Distribution

Histograms were used to study the distribution of:

* Age
* Number of floors
* Height percentage
* Area percentage

### Outlier Detection

Boxplots were used to identify potential outliers in numerical features.

### Correlation Analysis

A correlation heatmap was used to understand relationships between numerical features.

The top features correlated with `damage_grade` were also identified.

---

# Machine Learning Models

## 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

## 2. Decision Tree

Decision Tree was used to capture non-linear relationships between building characteristics and damage grade.

## 3. Random Forest

Random Forest combines multiple decision trees and generally provides more robust predictions than a single decision tree.

## 4. Gradient Boosting

Gradient Boosting was used to build an ensemble model sequentially and improve prediction performance.

## 5. XGBoost

XGBoost was used as the advanced boosting model and achieved the best overall performance among the evaluated models.

---

# Model Performance Comparison

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| **XGBoost**         | **0.7263** | **0.7270** | **0.7263** | **0.7183** |
| Random Forest       |     0.7161 |     0.7165 |     0.7161 |     0.7078 |
| Gradient Boosting   |     0.6813 |     0.6871 |     0.6813 |     0.6637 |
| Decision Tree       |     0.6459 |     0.6468 |     0.6459 |     0.6463 |
| Logistic Regression |     0.5883 |     0.5758 |     0.5883 |     0.5363 |

---

# Best Model

Based on the model comparison, **XGBoost** was selected as the final model.

### Performance

* **Accuracy:** 72.63%
* **Precision:** 72.70%
* **Recall:** 72.63%
* **F1 Score:** 71.83%

XGBoost achieved the highest overall performance among the evaluated models.

---

# Hyperparameter Tuning

GridSearchCV was used to find suitable XGBoost hyperparameters.

The parameters considered were:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`

The search was performed using **5-fold cross-validation**.

### Best Parameters

```text
learning_rate = 0.1
max_depth = 7
n_estimators = 200
subsample = 0.8
```

### Best Cross-Validation Accuracy

**72.53%**

---

# Suggestions to Seismologists

Based on the analysis, the following measures can help reduce earthquake-related building damage:

1. Promote earthquake-resistant building construction.
2. Strengthen vulnerable existing buildings.
3. Give special attention to buildings with weak structural characteristics.
4. Encourage the use of stronger and safer construction materials.
5. Improve foundation and structural design standards.
6. Conduct regular structural safety inspections.
7. Identify high-risk geographic regions and prioritize them for mitigation.
8. Improve enforcement of earthquake-resistant building codes.
9. Use building-risk prediction models for disaster preparedness.
10. Conduct public awareness programs on earthquake-safe construction and preparedness.

Machine Learning predictions can be used as a supporting tool for identifying buildings that may require additional structural assessment.

---

# Challenges Faced

### 1. Categorical Variables

Many features were categorical and could not be directly provided to traditional Machine Learning models.

**Solution:** One-Hot Encoding was used to convert categorical variables into numerical features.

### 2. Different Feature Scales

Numerical features had different ranges.

**Solution:** StandardScaler was used for models such as Logistic Regression.

### 3. Multi-Class Classification

The target variable contains three damage grades.

**Solution:** Multiple classification algorithms were evaluated using suitable multi-class metrics.

### 4. Model Selection

Different models produced different levels of performance.

**Solution:** Accuracy, Precision, Recall, and F1 Score were compared to select the best-performing model.

### 5. Hyperparameter Selection

The performance of XGBoost depends on suitable hyperparameter values.

**Solution:** GridSearchCV with 5-fold cross-validation was used to identify a suitable parameter combination.

---

# Conclusion

This project analyzed earthquake-affected building data and developed Machine Learning models to predict building damage grades.

Five classification models were evaluated. Among them, **XGBoost achieved the best performance with an accuracy of 72.63% and an F1 Score of 71.83%**.

Therefore, XGBoost was selected as the final model for earthquake damage prediction. The results can support risk assessment and disaster preparedness by helping identify building characteristics associated with higher levels of earthquake damage.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

# Repository Structure

```text
Earthquake-Damage-Prediction-ML/
│
├── Earthquake_Damage_Prediction_ML_Project.ipynb
├── train_values.csv
├── train_labels.csv
├── README.md
└── .gitignore
```

---

# Project Files

The main project analysis and Machine Learning implementation are available in:

**`Earthquake_Damage_Prediction_ML_Project.ipynb`**

The notebook contains data preprocessing, EDA, model building, model evaluation, hyperparameter tuning, final model selection, and conclusions.
