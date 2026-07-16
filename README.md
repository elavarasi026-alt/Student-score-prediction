Student Exam Score Predictor
A Python-based machine learning project that uses Linear Regression to predict a student's exam score based on their study habits, test preparation, and parental education level. It includes a trained predictive model, evaluation metrics, and an interactive command-line interface for real-time predictions.

Features
Machine Learning Pipeline: Preprocesses data, splits it into training and testing sets, and trains a scikit-learn Linear Regression model.

Model Evaluation: Evaluates performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

Interactive Command Line Interface: A continuous-input loop allowing users to input custom parameters and instantly get a predicted exam score.

How It Works
The model predicts the Exam_Score using three key features:

Hours Studied (Numeric)

Test Preparation (Binary: 0 for No, 1 for Yes)

Parental Education (Binary: 0 for No, 1 for Yes)
