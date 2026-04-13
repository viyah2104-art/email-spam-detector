
import pandas as pd

df = pd.read_csv("data/spam.csv")

print("📊 Dataset Info:")
print(df.head())
print("\nLabel Count:")
print(df['label'].value_counts())