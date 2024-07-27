import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load the dataset
file_path = '/Users/mounirahamad/Desktop/Real estate.csv'
df = pd.read_csv(file_path)

# Drop the 'No' and 'X1 transaction date' columns
df.drop(columns=['No', 'X1 transaction date'], inplace=True)

# Define features (X) and target variable (y)
X = df.drop(columns=['Y house price of unit area'])
y = df['Y house price of unit area']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model to a file using pickle
model_filename = 'real_estate_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
