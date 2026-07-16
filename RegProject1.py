import pandas as pa
import numpy as nu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df = pa.read_csv("student_scores.csv")

x = df[['Hours_Studied','Test_Preparation','Parental_Education']]
y = df['Exam_Score']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

print("X shape:", x.shape)
print("Y shape:", y.shape)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_pred,y_test)
rmse = nu.sqrt(mean_squared_error(y_pred,y_test))
r2 = r2_score(y_pred,y_test)

print("Model Evaluation:")
print(round(mae, 2))
print(round(rmse, 2))
print(round(r2, 2))

result = pa.DataFrame({"Actual Score": y_test , "Predicted Score" : y_pred})
print(result)

#User Input

while True:
    try:
        hours_studied = float(input("Enter hours studied: "))
        test_preparation = int(input("Enter test preparation (0 for no, 1 for yes): "))
        parental_education = int(input("Enter parental education (0 for no, 1 for yes): "))
        
        edu = [[hours_studied, test_preparation, parental_education]]
        predicted_score = model.predict(edu)
        print("Predicted Exam Score:", round(predicted_score[0], 2))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        continue
    
    another = input("\nPredict another student? (y/n): ").strip().lower()
    if another == 'n' or another == 'no':
        print("Exiting. Goodbye!")
        break
