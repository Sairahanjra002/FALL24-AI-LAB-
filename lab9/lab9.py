# ==========================================
# AI LAB 9: DATA EXPLORATION - LAPTOP DATASET
# ==========================================
import pandas as pd
import numpy as np

# 1. Load the Dataset
filename = 'laptop_data.csv'
# Attempt to read with different encodings in case of special characters
try:
    df = pd.read_csv(filename, encoding='latin-1')
except:
    df = pd.read_csv(filename)

print("--- Data Loaded Successfully ---\n")

# 2. View Top 5 Rows
print("Top 5 Rows:")
print(df.head())

# 3. View Bottom 5 Rows
print("\nBottom 5 Rows:")
print(df.tail())


# 4. Check Dataset Dimensions
rows, cols = df.shape
print(f"\nNumber of Rows: {rows}")
print(f"Number of Columns: {cols}")

# 5. Check for Null/Missing Values
print("\nNull Values per Column (Before Cleaning):")
print(df.isnull().sum())

# 6. Fill Missing Values
# Logic: Fill text columns with Mode (Most Frequent) and numerical columns with Mean
for col in df.columns:
    if df[col].dtype == 'object':
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].mean())

print("\nMissing values have been filled.")

# 7. Verify Data Types
print("\nColumn Data Types:")
print(df.dtypes)