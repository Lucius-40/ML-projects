SMS Spam Checker Project Documentation

Overview:
This project implements a machine learning pipeline to classify SMS messages as spam or not spam. It uses a dataset with two columns: the label (spam/ham) and the message content. The workflow includes feature extraction, data cleaning, vectorization, model training, evaluation, and visualization.

Steps:
1. **Data Loading**: The dataset is loaded from a tab-separated text file. Labels are converted to binary (1 for spam, 0 for ham).

2. **Feature Extraction**: Custom features are extracted from the message content, including:
   - Uppercase letter ratio
   - Exclamation and question mark counts
   - Special symbol count
   - URL count

3. **Data Cleaning**: Messages are cleaned using regular expressions, converted to lowercase, stemmed, and filtered to remove stopwords.

4. **Vectorization**: Cleaned text is vectorized using TF-IDF with n-grams. Custom features are combined with the vectorized data.

5. **Scaling and Splitting**: The combined feature set is split into training and test sets.

6. **Model Training**: A logistic regression model is trained on the training data.

7. **Evaluation**: Accuracy is measured on both training and test sets.

8. **Prediction Interface**: The model predicts the label for new messages and prints the result.

9. **Visualization**: Scatter plots are used to visualize the relationship between capitalization and special symbol usage in spam vs. non-spam messages.

10. **Insights**: Spam messages tend to have higher capitalization and more special symbols/URLs.

Dependencies:
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- seaborn
- scipy

Usage:
Run the notebook `spamCheck.ipynb` step by step to reproduce the analysis and results. Ensure all dependencies are installed and the dataset `smsSpam.txt` is present in the project directory.
