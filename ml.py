from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd

# Load the uploaded dataset
file_path = 'Food_Delivery_Times.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
#data.head(), data.info()

# Drop unnecessary column and separate features and target
data = data.drop(columns=["Order_ID"])
X = data.drop(columns=["Delivery_Time_min"])
y = data["Delivery_Time_min"]

# Identify categorical and numerical columns
categorical_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
numerical_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]

# Handle missing values and encode categorical data
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the regression pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(mse)
import joblib

# Assuming `model` is your trained machine learning model
model_file_path = "delivery_time_model.pkl"  # File path where the model will be saved

# Save the model to a file
joblib.dump(model, model_file_path)
