import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read the dataset
data = pd.read_csv("task 3/advertising.csv")

# Display the first few rows of the dataset
print(data.head())

# Visualize the relationships between features and target variable
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=0.7)
plt.show()

# Separate features (X) and target variable (y)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualize the actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.show()

# Predicting sales for a new set of advertising expenditures
new_data = pd.DataFrame({'TV': [200], 'Radio': [40], 'Newspaper': [60]})
predicted_sales = model.predict(new_data)
print("Predicted sales for new advertising expenditures:", predicted_sales)

# Explanation:
# In businesses that offer products or services, the role of a Data Scientist is crucial for predicting future sales.
# They utilize machine learning techniques in Python to analyze and interpret data, allowing them to make informed decisions regarding advertising costs.
# By leveraging these predictions, businesses can optimize their advertising strategies and maximize sales potential.
# Let's embark on the journey of sales prediction using machine learning in Python.
