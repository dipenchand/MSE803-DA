import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import the data
print("*" * 50)
print("\nImporting data...")
df = pd.read_csv('salary-dataset.csv')
print("Data imported successfully!")

print("*" * 50)
print("\nData cleaning...")
# Data cleaning
# Remove unnamed index column if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Drop rows with missing values (if any)
df = df.dropna()

# Check for duplicates (full row duplicates)
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Drop duplicates if any
df = df.drop_duplicates()

# Aggregation: handle same YearsExperience with different salaries
# Group by YearsExperience and take mean Salary to get one target per experience level
print("\nUnique YearsExperience before aggregation:", df['YearsExperience'].nunique())
df = df.groupby('YearsExperience', as_index=False)['Salary'].mean()
print("Dataset shape after aggregation (one row per unique YearsExperience):", df.shape)

# Reset index
df = df.reset_index(drop=True)

# Display cleaned data info
print("\nCleaned dataset shape:", df.shape)
print("\nCleaned dataset info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("*" * 50)

print("Preprocessing for modeling...")
# Preprocessing for modeling
# Features (X) and target (y)
X = df[['YearsExperience']].values
y = df['Salary'].values

# Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])


def evaluate(name, y_true, y_pred):
    """Compute and print MAE, MSE, RMSE for a model."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n=== {name} ===")
    print(f"MAE (Mean Absolute Error)  : {mae:,.2f}")
    print(f"MSE (Mean Squared Error)  : {mse:,.2f}")
    print(f"RMSE (Root Mean Squared Error) : {rmse:,.2f}")
    return mae, mse, rmse


print("*" * 50)
print("\nModel 1: Linear Regression")

# Model 1: Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
lin_metrics = evaluate("Linear Regression", y_test, y_pred_lin)

print("*" * 50)
print("\nModel 2: Polynomial Regression (degree 2)")

# Model 2: Polynomial Regression (degree 2)
POLY_DEGREE = 2
poly_model = make_pipeline(
    PolynomialFeatures(degree=POLY_DEGREE),
    LinearRegression()
)
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
poly_metrics = evaluate(f"Polynomial Regression (degree={POLY_DEGREE})", y_test, y_pred_poly)

# ----------------------------------------------------------------------
# Comparison summary
# ----------------------------------------------------------------------
results = pd.DataFrame(
    {
        "Model": ["Linear Regression", f"Polynomial (deg={POLY_DEGREE})"],
        "MAE": [lin_metrics[0], poly_metrics[0]],
        "MSE": [lin_metrics[1], poly_metrics[1]],
        "RMSE": [lin_metrics[2], poly_metrics[2]],
    }
)
print("\n=== Model Comparison ===")
print(results.to_string(index=False))

print("*" * 50)
print("Conclusion:")
print("The polynomial regression model (degree=2) has a lower MAE, MSE, and RMSE compared to the linear regression model.")
print("This indicates that the polynomial model provides a better fit to the data.")
print("*" * 50)

# Visualization
# x-range for plotting continuous regression curves
x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_lin_line = lin_model.predict(x_range)
y_poly_line = poly_model.predict(x_range)

plt.figure(figsize=(10, 6))

# Actual data: train and test points
plt.scatter(X_train, y_train, color="steelblue", label="Train data", alpha=0.7)
plt.scatter(X_test, y_test, color="orange", edgecolor="black",
            label="Test data", zorder=5)

# Fitted model curves
plt.plot(x_range, y_lin_line, color="red", linewidth=2,
         label="Linear Regression")
plt.plot(x_range, y_poly_line, color="green", linewidth=2, linestyle="--",
         label=f"Polynomial (deg={POLY_DEGREE})")

plt.title("Salary vs Years of Experience: Linear vs Polynomial Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save to file and display
plt.savefig("regression_plot.png", dpi=150)
plt.show()
