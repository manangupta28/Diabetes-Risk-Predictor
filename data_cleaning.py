import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler 

dataset = r'C:\Users\hatak\Downloads\diabetes_prediction_dataset.csv.zip'

df = pd.read_csv(dataset)

diabetes_count = df[df['diabetes'] == 1].shape[0]

df['gender'] = df['gender'].str.title()
df = df[(df['bmi'] >= 10) & (df['bmi'] <= 50)]
df = df[df['age'] >= 10]
df.to_csv('clenaed_dataset.csv',index= False)

X = df.drop(columns=['diabetes'])  # Features
y = df['diabetes']  # Target

X['gender'] = X['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})


X = pd.get_dummies(X, columns=['smoking_history'], drop_first=True)


scaler = MinMaxScaler()
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


X.to_csv('prepared_features.csv', index=False)
y.to_csv('prepared_target.csv', index=False)

print("Data preparation complete! Prepared files are saved as 'prepared_features.csv' and 'prepared_target.csv'.")




