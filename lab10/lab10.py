# ==========================================================
# AI LAB 10: DATA PRE-PROCESSING & FEATURE ENGINEERING
# (Laptop Dataset)
# ==========================================================
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore') 

# --- 1. Data Loading ---
print("######################################################")
print("## 1. DATA LOADING AND INITIAL CLEANING ##")
print("######################################################")

try:
    df = pd.read_csv('laptop_data.csv', encoding='latin-1')
except:
    df = pd.read_csv('laptop_data.csv')

# Drop unnecessary columns (ID columns)
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
    
print(f"--> Dataset Loaded. Remaining Columns: {df.shape[1]}\n")

# --- 2. Data Cleaning & Type Conversion ---
# Convert 'Ram' column to integer after removing 'GB'
if df['Ram'].dtype == 'object':
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    print("--> 'Ram' cleaned and converted to integer.")

# Convert 'Weight' column to float after removing 'kg'
if df['Weight'].dtype == 'object':
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
    print("--> 'Weight' cleaned and converted to float.")

# --- 3. Null Value Handling ---
print("\n######################################################")
print("## 2. NULL VALUE HANDLING ##")
print("######################################################")

for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == 'object':
            # Fill categorical columns with Mode (Most Frequent)
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            # Fill numerical columns with Mean
            df[col] = df[col].fillna(df[col].mean())
            
print("--> Null values filled using Mode (Text) and Mean (Numbers).")

# Final verification check
print("\n--- Final Null Value Check ---")
if df.isnull().sum().sum() == 0:
    print("Zero missing values remaining. Data is Clean.")
else:
    print("WARNING: Some missing values remain.")


# --- 4. Final Checks (Exploration) ---
print("\n######################################################")
print("## 3. FINAL EXPLORATION CHECKS ##")
print("######################################################")

# Print rows and columns / shape of dataframe
print(f"Dataset Final Shape: {df.shape}")

# Check datatypes of all columns
print("\n--- Column Data Types ---")
print(df.dtypes)

# Check unique values of a categorical column
print("\n--- Unique Values Check (Example: Company) ---")
print(f"Company Unique Count: {df['Company'].nunique()}")


# --- 5. Splitting and Encoding ---
print("\n######################################################")
print("## 4. FEATURE SELECTION AND ENCODING ##")
print("######################################################")

target_col = 'Price'

# X = Features (All columns except target)
X = df.drop(target_col, axis=1)
# Y = Target
Y = df[target_col]

print(f"X (Features) Shape: {X.shape}")
print(f"Y (Target) Shape: {Y.shape}")

# Converting Object columns into Int columns (Factorization)
cat_columns = X.select_dtypes(['object']).columns

if len(cat_columns) > 0:
    # Use Factorize to convert unique strings into unique integers
    X[cat_columns] = X[cat_columns].apply(lambda x: pd.factorize(x)[0])
    
    # Safety Check: Replace any negative values (-1 from factorize error) with 0
    if (X < 0).any().any():
        X[X < 0] = 0
    
    print(f"Encoded categorical columns: {list(cat_columns)}")
    
print("\n--- Pre-processing Completed ---")
print("Sample Features (X.head()):")
print(X.head())