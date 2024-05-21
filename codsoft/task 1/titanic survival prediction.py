# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:24:38 2024

@author: dirav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Titanic dataset
#titanic_data=pd.read_csv("C:/Users/dirav/codsoft/task 1/Titanic-Dataset.csv")
titanic_data=pd.read_csv("task 1/Titanic-Dataset.csv")

#titanic_data=pd.read_csv('/kaggle/input/titanic-dataset/Titanic-Dataset.csv')
# Data preprocessing
# Drop irrelevant columns
titanic_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# Fill missing values in 'Age' column with the mean age
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
# Convert categorical variables into dummy/indicator variables
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'])

# Split the data into features (X) and target (y)
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
logreg_model = LogisticRegression(max_iter=1500)

logreg_model.fit(X_train, y_train)

# Initialize and train the random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Initialize and train the decision tree classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
logreg_pred = logreg_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

# Evaluate the models
models = {'Logistic Regression': logreg_pred, 'Random Forest': rf_pred, 'Decision Tree': dt_pred}
for name, pred in models.items():
    accuracy = accuracy_score(y_test, pred)
    print(f"Accuracy of {name}: {accuracy}")
    print(f"Classification report of {name}:")
    class_rep = classification_report(y_test, pred, target_names=['Not Survived', 'Survived'])
    print(class_rep)

# Confusion matrix for logistic regression model
conf_matrix = confusion_matrix(y_test, logreg_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Histogram of passenger ages
plt.figure(figsize=(8, 6))
sns.histplot(titanic_data['Age'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Passenger Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Bar plot of passenger survival counts
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=titanic_data, palette='pastel')
plt.title('Survival Counts')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()
