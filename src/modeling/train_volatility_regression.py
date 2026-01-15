import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.modeling.features import FEATURES
from src.modeling.split import temporal_split

# Load
df = pd.read_csv("data/synthetic/member_month_with_volatility.csv")

# Keep only rows where target is defined
df = df.dropna(subset=["volatility_score"])

# Temporal split (out-of-time validation)
train, test, cutoff_ym = temporal_split(df)
print("Temporal split cutoff ym:", cutoff_ym)
print("Train rows:", len(train), "Test rows:", len(test))


X_train = train[FEATURES]
y_train = train["volatility_score"]

X_test = test[FEATURES]
y_test = test["volatility_score"]

# Industry pattern: preprocessing + model in one pipeline
model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", LinearRegression()),
])

model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = mse ** 0.5

print("Volatility Regression Results")
print("-----------------------------")
print(f"MAE:  {mae:.6f}")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"y_test mean: {y_test.mean():.6f}  std: {y_test.std():.6f}  min: {y_test.min():.6f}  max: {y_test.max():.6f}")


# Quick diagnostic: how many NaNs were in features?
nan_rate = X_train.isna().mean().sort_values(ascending=False)
top_nan = nan_rate[nan_rate > 0].head(10)
if len(top_nan) > 0:
    print("\nTop missing feature rates (train):")
    print(top_nan)
else:
    print("\nNo missing values found in features after filtering (unexpected).")
