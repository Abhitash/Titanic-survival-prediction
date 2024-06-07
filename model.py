import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
test_data = pd.read_csv('tit_test.csv')
train_data = pd.read_csv('tit_train.csv')

# Initial findings from the data
print(train_data.head())
print(train_data.info())
print("THE NUMBER OF NULL DATA IS")
print(train_data.isnull().sum())

# Plotting the graphs for the data
sns.countplot(x='Survived', data=train_data)
plt.title('Count of Survived Passengers')
plt.show()

# Plotting with survived vs sex
sns.countplot(x='Survived', hue='Sex', data=train_data)
plt.title('Count of Survived Passengers vs Sex')
plt.show()

# Plotting with survived vs age
sns.countplot(x='Survived', hue='Age', data=train_data)
plt.title('Count of Survived Passengers vs Age')
plt.show()

# Check for skewness
print("Skewness of Age:", train_data['Age'].skew())

# Drops the duplicates
train_data = train_data.drop_duplicates()

# Fill missing values in 'Age' with median
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
print("TOTAL AMOUNT OF MISSING VALUES AFTER FILLING AGE:")
print(train_data.isnull().sum())

# Drop 'Cabin' column due to many missing values
train_data.drop(columns=['Cabin'], inplace=True)

# Fill missing values in 'Embarked' with the mode
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Final check for missing values
print("TOTAL AMOUNT OF MISSING VALUES AFTER CLEANING:")
print(train_data.isnull().sum())

# Formation of new features
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)

# Encode categorical variables
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)

# Select features
features = ['Sex', 'Age', 'FamilySize', 'IsAlone']
X = train_data[features]
y = train_data['Survived']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)

# Evaluate the model
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred))
