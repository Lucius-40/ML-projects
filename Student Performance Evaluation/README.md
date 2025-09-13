Student Performance Evaluation - Documentation

Overview:
This notebook demonstrates a simple machine learning workflow to predict student performance index using various features from a dataset. The workflow includes data preprocessing, visualization, model training, evaluation, and prediction visualization.

Workflow Steps:

1. **Dependencies**
	- Import required libraries: pandas, seaborn, matplotlib, scikit-learn.

2. **Data Import and Preprocessing**
	- Load the dataset (`Student_Performance.csv`) into a pandas DataFrame.
	- Convert categorical features (e.g., 'Extracurricular Activities') to numerical values for modeling.

3. **Exploratory Data Analysis**
	- Display data columns, sample rows, and data types.
	- Visualize feature correlations using a heatmap.
	- Plot scatter plots to show relationships between significant features and the performance index.

4. **Data Splitting**
	- Select relevant features for modeling.
	- Split the data into training and test sets using `train_test_split`.

5. **Feature Scaling**
	- Standardize features using `StandardScaler` to improve model performance.

6. **Model Training**
	- Train a linear regression model on the scaled training data.

7. **Model Evaluation**
	- Predict performance index for the test set.
	- Calculate regression metrics: Mean Squared Error (MSE) for both train and test sets.

8. **Prediction Visualization**
	- Visualize predicted vs true performance index using seaborn scatter plots.
	- Display a DataFrame comparing predictions and actual values.

Usage Notes:
- The notebook is designed for regression tasks, not classification.
- Feature scaling is performed only on numerical features.
- Visualization helps assess model fit and prediction quality.

End of Documentation
