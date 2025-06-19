import numpy as np
import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the CSV data into a Pandas DataFrame
file_path = "heart_disease_data.csv"  # Use a relative path
heart_data = pd.read_csv(file_path)

# Display first and last 5 rows of the dataset
print(heart_data.head())
print(heart_data.tail())

# Dataset shape
print("Shape of the dataset:", heart_data.shape)

# Dataset information
print("Dataset Info:")
print(heart_data.info())

# Checking for missing values
print("Missing values in each column:")
print(heart_data.isnull().sum())

# Statistical description
print("Statistical description of the dataset:")
print(heart_data.describe())

# Ensure the target column exists
if 'target' not in heart_data.columns:
    raise ValueError("The dataset does not contain a 'target' column.")

# Checking the distribution of the target variable
print("Distribution of target variable:")
print(heart_data['target'].value_counts())

# Splitting features (X) and target (Y)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Splitting the data into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print(f"Shapes -> X: {X.shape}, X_train: {X_train.shape}, X_test: {X_test.shape}")

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f'Accuracy on Training data: {training_data_accuracy:.2f}')

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f'Accuracy on Test data: {test_data_accuracy:.2f}')

# Save the trained model as 'model.pkl'
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model saved successfully as 'model.pkl'")
