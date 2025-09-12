# Linear Regression from Scratch (Numpy)

## Overview
This project demonstrates how to implement linear regression using only numpy and pandas, without relying on scikit-learn's built-in models. The approach covers the cost function, gradient computation, and gradient descent optimization, and visualizes the results.

## Steps

### 1. Data Preparation
- Synthetic regression data is generated using `sklearn.datasets.make_regression`.
- Features are stored in a pandas DataFrame for easy manipulation.

### 2. Cost Function
- The cost function used is **Mean Squared Error (MSE)**:
  $$
  J(W, b) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - (\mathbf{x}_i \cdot W + b))^2
  $$
- This measures the average squared difference between actual and predicted values.

### 3. Gradient Computation
- Gradients are calculated for both weights (`W`) and bias (`b`):
  - $$
    \text{grad}_W = \frac{1}{m} X^T \cdot (XW + b - Y)
    $$
  - $$
    \text{grad}_b = \frac{1}{m} \sum (XW + b - Y)
    $$
- These gradients are used to update the parameters during training.

### 4. Gradient Descent
- The model parameters are updated iteratively using gradient descent:
  - $$
    W := W - \alpha \cdot \text{grad}_W
    $$
  - $$
    b := b - \alpha \cdot \text{grad}_b
    $$
- The learning rate (`lr`) controls the step size.

### 5. Training
- The model is trained for a specified number of iterations.
- Cost is printed and logged every 1000 iterations to monitor convergence.

### 6. Evaluation
- After training, the learned weights and bias are used to make predictions.
- A scatter plot compares actual vs predicted values, with a red dashed line indicating perfect predictions.

## Usage

- **Functions:**
  - `cost_function(X, Y, W, b)`: Computes the MSE cost.
  - `gradient(X, Y, W, b)`: Computes gradients for weights and bias.
  - `gradient_descendt(X, Y, W, b, iterations, lr)`: Runs gradient descent optimization.

- **Workflow:**
  1. Prepare data (`X_df`, `Y`).
  2. Initialize weights and bias.
  3. Train using `gradient_descendt`.
  4. Visualize results.

## Notes
- The implementation is vectorized for efficiency.
- The model works for any number of features.
- Visualization helps assess model fit and error distribution.

## Example

```python
X, Y = make_regression(n_samples=1000, n_features=6, noise=5, random_state=42)
X_df = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(X.shape[1])])
w = np.zeros(X_df.shape[1])
W_set, bs = gradient_descendt(X_df, Y, w, 0, 10000)
```

---

This documentation describes the linear regression implementation and usage. For details, see the notebook `Linear_Regression.ipynb`.