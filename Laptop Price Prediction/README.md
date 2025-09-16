# Laptop Price Prediction

A machine learning project for predicting laptop prices based on their specifications. This project is designed for learning purposes and demonstrates a robust ML pipeline using custom transformers, feature engineering, and model evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Model Evaluation](#model-evaluation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Data Exploration](#data-exploration)

## Project Overview
This project aims to predict the price of laptops using various features such as processor, RAM, storage type (SSD, HDD, Flash), GPU, and more. The workflow includes data cleaning, feature extraction, preprocessing, model training, and evaluation. The project is intended for educational purposes and to demonstrate best practices in building ML pipelines.

## Dataset
- **Source:** Kaggle (see [smsSpam.txt](../smsSpam.txt) for reference to data usage)
- **Description:** The dataset contains laptop specifications and their corresponding prices.
- **Note:** The dataset is not included in this repository due to licensing restrictions. Please download it directly from Kaggle.

## Features
- Brand, Model, Processor, RAM, Storage (SSD, HDD, Flash), GPU, OS, Display, Weight, and more.
- Custom feature extraction for storage types (SSD, HDD, Flash).
- Handling of missing values, categorical encoding, and scaling.

## Pipeline Architecture
- **Custom Transformers:**
  - `ColumnSelector`: Selects relevant columns.
  - `DataTypeFixer`: Ensures correct data types.
  - `FeatureExtract`: Extracts features like SSD, HDD, Flash from storage info.
  - `SkewFixer`: Fixes skewness in numerical features.
  - `Scaler`: Scales numerical features.
  - `OneHot`: Encodes categorical variables.
  - `DropColumnsTransformer`: Drops unnecessary columns.
- **Model Selection:**
  - Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost, CatBoost, SVR, KNN, and more.
- **Evaluation Metrics:**
  - MAE, RMSE, R², and MAPE (Mean Absolute Percentage Error).

## Model Evaluation
Models are evaluated using cross-validation with the following metrics:
- **MAE (Mean Absolute Error):** Measures average absolute errors.
- **RMSE (Root Mean Squared Error):** Penalizes larger errors.
- **R² (R-squared):** Proportion of variance explained.
- **MAPE (Mean Absolute Percentage Error):** Expresses error as a percentage.

## How to Run
1. **Install Requirements:**
   - Python 3.8+
   - Install dependencies:
     ```bash
     pip install pandas numpy scikit-learn matplotlib seaborn xgboost catboost
     ```
2. **Download Dataset:**
   - Download the laptop price dataset from Kaggle and place it in the project directory.
3. **Run the Notebook:**
   - Open `clean_and_predict.ipynb` in Jupyter Notebook or VS Code.
   - Run all cells to execute the pipeline and evaluate models.

## Results
- The pipeline compares multiple regression models and reports their performance using cross-validation.
- Feature engineering and robust preprocessing improve model accuracy.
- See the notebook for detailed results and visualizations.

## Data Exploration
Before building the machine learning pipeline, the dataset was explored to understand its structure and key characteristics. The following steps were performed:

- **Loading the Data:**
  - The dataset was loaded into a pandas DataFrame for analysis.
- **Initial Inspection:**
  - Displayed the first few rows to get an overview of the data.
  - Checked the shape (number of rows and columns).
- **Summary Statistics:**
  - Used `df.describe()` to view mean, standard deviation, min, max, and quartiles for numerical columns.
  - Used `df.info()` to check data types and missing values.
- **Missing Values:**
  - Identified columns with missing values and their counts.
- **Categorical Features:**
  - Listed unique values for key categorical columns (e.g., Brand, Processor, GPU, OS).
- **Target Variable:**
  - Analyzed the distribution of the target variable (Price) using histograms and boxplots.
- **Correlation Analysis:**
  - Computed correlation matrix to identify relationships between numerical features and the target.
- **Visualization:**
  - Plotted distributions of important features (e.g., RAM, Storage, Price).
  - Used pairplots and heatmaps for deeper insights.

These steps provided a solid understanding of the data, informed feature engineering, and guided the preprocessing pipeline.