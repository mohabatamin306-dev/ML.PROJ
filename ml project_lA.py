# house_price_analysis.py

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
df = pd.read_csv("house_price_regression_dataset.csv")

# --------------------------------------------------
# 2. Print dataset info
# --------------------------------------------------
print("Dataset shape:", df.shape)
print("\nDataset head:")
print(df.head())
print("\nColumns and data types:")
print(df.dtypes)

# --------------------------------------------------
# 3. Check for NaN values and drop them
# --------------------------------------------------
print("\nChecking for NaN values:")
print(df.isna().sum())

df.dropna(inplace=True)
print("After dropping NaNs, new shape:", df.shape)

# --------------------------------------------------
# 4. Convert categorical features to numbers
# --------------------------------------------------
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"Encoded {col} to numeric.")

# --------------------------------------------------
# 5. Visualizations
# --------------------------------------------------
target_col = df.columns[-1]

# Scatter plots
for col in df.columns[:-1]:
    plt.figure(figsize=(6,4))
    plt.scatter(df[col], df[target_col])
    plt.xlabel(col)
    plt.ylabel(target_col)
    plt.title(f"{col} vs {target_col}")
    plt.show()

# Bar plots (small categorical-like features)
for col in df.columns:
    if df[col].dtype in ['int64', 'int32'] and len(df[col].unique()) < 20:
        plt.figure(figsize=(6,4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f"{col} value counts")
        plt.show()

# Boxplot for outliers
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.title("Boxplot for all features")
plt.show()

# --------------------------------------------------
# 6. Remove outliers using IQR
# --------------------------------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print("After removing outliers, new shape:", df.shape)

# --------------------------------------------------
# 7. Scale features
# --------------------------------------------------
X = df.drop(target_col, axis=1)
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 8. Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 9. Train Linear Regression model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# 10. Test the model
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# 11. Compare predictions with actual values
# --------------------------------------------------
comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print("\nActual vs Predicted (first 5 rows):")
print(comparison.head())

# Plot predictions vs actual
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.show()

# --------------------------------------------------
# 12. Evaluate the model
# --------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# --------------------------------------------------
# 13. USER INPUT HOUSE PRICE PREDICTION
# --------------------------------------------------
print("\n===================================")
print("HOUSE PRICE PREDICTION (Your Input)")
print("===================================")

print("\nModel expects these features:")
print(list(X.columns))

user_features = {}

for feature in X.columns:
    value = float(input(f"Enter {feature}: "))
    user_features[feature] = value

# Convert user input to DataFrame
user_df = pd.DataFrame([user_features])

# Scale user input using the SAME scaler
user_scaled = scaler.transform(user_df)

# Predict price
predicted_price = model.predict(user_scaled)

print("\n🏠 Predicted House Price:")
print(predicted_price[0])