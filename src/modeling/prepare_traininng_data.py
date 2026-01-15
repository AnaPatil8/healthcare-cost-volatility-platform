import pandas as pd

df = pd.read_csv("data/synthetic/member_month_with_volatility.csv")

# Only rows with valid volatility labels
df_model = df.dropna(subset=["volatility_score"])

print("Total rows:", len(df))
print("Training-eligible rows:", len(df_model))
print("Percent usable:", round(len(df_model) / len(df) * 100, 2))

print("\nSpike rate:")
print(df_model["spike_flag_member"].mean())
