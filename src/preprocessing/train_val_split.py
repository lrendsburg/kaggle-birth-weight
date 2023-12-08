import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df_train_full = pd.read_csv("datasets/train_full.csv")
    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=42)
    df_train.to_csv("datasets/train.csv", index=False)
    df_val.to_csv("datasets/val.csv", index=False)
