# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("task 2/IRIS.csv")
#data =pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
# Separate features and target variable
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]  # Only considering sepal and petal measurements
y = data['species']

# Encoding the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Support Vector Classifier
svm_classifier = SVC(kernel='rbf', random_state=42)  # Using radial basis function kernel

# Train the Support Vector Classifier
svm_classifier.fit(X_train, y_train)

# Predictions with Support Vector Classifier
y_pred_svm = svm_classifier.predict(X_test)

# Calculate accuracy for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Support Vector Machine Accuracy:", accuracy_svm)

# Classification report for SVM
print("Support Vector Machine Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

# Initialize Logistic Regression Classifier
logreg_classifier = LogisticRegression(random_state=42, max_iter=1000)  

# Train the Logistic Regression Classifier
logreg_classifier.fit(X_train, y_train)

# Predictions with Logistic Regression Classifier
y_pred_logreg = logreg_classifier.predict(X_test)

# Calculate accuracy for Logistic Regression
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("\nLogistic Regression Accuracy:", accuracy_logreg)

# Classification report for Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg, target_names=le.classes_))

# Initialize Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree Classifier
dt_classifier.fit(X_train, y_train)

# Predictions with Decision Tree Classifier
y_pred_dt = dt_classifier.predict(X_test)

# Calculate accuracy for Decision Tree Classifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("\nDecision Tree Classifier Accuracy:", accuracy_dt)

# Classification report for Decision Tree Classifier
print("Decision Tree Classifier Classification Report:")
print(classification_report(y_test, y_pred_dt, target_names=le.classes_))

