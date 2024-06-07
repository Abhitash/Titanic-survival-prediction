import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def add_elements():
    print("ENTER THE ANSWER FOR STRING AS 1 OR 0 ")
    gen = input("ENTER THE GENDER (Male: 0, Female: 1): ")
    age = int(input("ENTER THE AGE: "))
    hypertension = int(input("ENTER THE HYPERTENSION (0 or 1): "))
    heart_disease = int(input("ENTER THE HEART DISEASE (0 or 1): "))
    smoking_history = int(input("ENTER THE SMOKING HISTORY (0 or 1): "))
    bmi = float(input("ENTER THE BMI: "))
    HbA1c_level = float(input("ENTER THE HbA1c LEVEL: "))
    blood_glucose_level = float(input("ENTER THE BLOOD GLUCOSE LEVEL: "))

    # Create a dictionary with the input values
    x_new = {
        'gender': 1 - int(gen),  # Convert to Male: 1, Female: 0
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': np.log(bmi) if bmi > 0 else np.nan,  # Apply log transformation to bmi
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level
    }
    return x_new

data = pd.read_csv("diabetes_prediction_dataset.csv")

# initial data exploration
print(data.head())
print(data.info())

print(data.isnull().sum())
# no null data is present in any column

sns.countplot(x='diabetes',hue='smoking_history',data=data)
plt.show()

sns.countplot(x='diabetes',hue='age',data=data)
plt.show()

sns.countplot(x='diabetes',hue='gender',data=data)
plt.show()

print("skewness of age is",data['age'].skew())

print("skewness of bmi",data['bmi'].skew())
data['bmi'] = data['bmi'].apply(lambda x: np.log(x) if x > 0 else np.nan)
print("skewness of bmi",data['bmi'].skew())
data = pd.get_dummies(data, columns=['smoking_history', 'age', 'gender'], drop_first=True)
# dropping duplicate value if exits
data = data.drop_duplicates()

x = data.drop(columns='diabetes', axis=1)
y = data['diabetes']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred_train = model.predict(x_train)

print("accuracy of seen data that is x_train", accuracy_score(y_train, y_pred_train))

y_pred_test = model.predict(x_test)
print("Accuracy of y prediction on test data is that is unseen data", accuracy_score(y_test, y_pred_test))

# Function to preprocess input data
def preprocess_input(input_data):
    # Apply log transformation to BMI
    input_data['bmi'] = input_data['bmi'].apply(lambda x: np.log(x) if x > 0 else np.nan)
    
    # Ensure that all feature names match those seen during training
    # If any features are missing, add them with default values
    expected_features = x_train.columns
    input_data = input_data.reindex(columns=expected_features, fill_value=0)
    
    return input_data

# Function to add elements
input_data = add_elements()

# Convert input data into DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess input data
input_df_processed = preprocess_input(input_df)

# Make predictions with the trained model
prediction = model.predict(input_df_processed)

# Output the prediction
print("THE OUTCOME FOR THE GIVEN FEATURES IS", prediction[0])