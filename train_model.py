import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib 


X = pd.read_csv('prepared_features.csv')
y = pd.read_csv('prepared_target.csv').values.ravel()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print(f'Model Accuracy: {accuracy_score(y_test, y_pred):.2f}')

joblib.dump(model, 'diabetes_risk_model.pkl') 
