import joblib
import numpy as np


model = joblib.load('diabetes_risk_model.pkl')

def get_float_input(prompt):
    """Ensures user enters a valid float."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("❌ Invalid input! Please enter a valid number.")

def get_int_input(prompt, valid_options=None):
    """Ensures user enters a valid integer, optionally restricted to a set of values."""
    while True:
        try:
            value = int(input(prompt))
            if valid_options and value not in valid_options:
                print(f"❌ Invalid choice! Please enter one of {valid_options}.")
            else:
                return value
        except ValueError:
            print("❌ Invalid input! Please enter a valid number.")


print("Enter patient details to predict diabetes risk:")

age = get_float_input("Age: ")
gender = input("Gender (Male/Female/Other): ").strip().title()


gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
gender = gender_mapping.get(gender, 2)  # Default to 'Other' if invalid input

bmi = get_float_input("BMI: ")
hba1c = get_float_input("HbA1c Level: ")
glucose = get_float_input("Blood Glucose Level: ")

hypertension = get_int_input("Hypertension (0 for No, 1 for Yes): ", [0, 1])
heart_disease = get_int_input("Heart Disease (0 for No, 1 for Yes): ", [0, 1])


print("\nSmoking History Options: ")
print("1 - Current Smoker\n2 - Ever Smoked\n3 - Former Smoker\n4 - Never Smoked\n5 - Not Currently Smoking")

smoking_choice = get_int_input("Choose an option (1-5): ", [1, 2, 3, 4, 5])


smoking_features = [0, 0, 0, 0, 0]
smoking_features[smoking_choice - 1] = 1  # Set selected smoking category to 1


input_data = np.array([[gender, age, hypertension, heart_disease, bmi, hba1c, glucose] + smoking_features])


prediction = model.predict(input_data)[0]


print("\n⚠️ HIGH risk of diabetes." if prediction == 1 else "\n✅ LOW risk of diabetes.")
