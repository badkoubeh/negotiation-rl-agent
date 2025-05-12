import pandas as pd
from sklearn.model_selection import train_test_split

# Load raw dataset
df = pd.read_parquet("data/raw/ioi_data.parquet")

# Encode categorical variables
df = pd.get_dummies(df, columns=["side", "status"])

# Split into train/test/inference
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, infer_df = train_test_split(test_df, test_size=0.5, random_state=42)

# Save splits as Parquet files
train_df.to_parquet("data/processed/ioi_train.parquet", index=False)
test_df.to_parquet("data/processed/ioi_test.parquet", index=False)
infer_df.to_parquet("data/processed/ioi_inference.parquet", index=False)

