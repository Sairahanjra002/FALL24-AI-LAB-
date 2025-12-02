# =======================================================
# AI LAB 11: REGRESSION & PLOTS - LAPTOP DATASET (PRICE PREDICTION)
# Simple print output for clean terminal display.
# =======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# --- REGRESSION MODEL IMPORTS ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# --- 1. DATA PREPARATION ---
print("--- 1. DATA PREPARATION ---")

# Data Loading and Initial Cleaning
try:
    df = pd.read_csv('laptop_data.csv', encoding='latin-1')
except:
    df = pd.read_csv('laptop_data.csv')

# Clean and Convert Columns
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
if 'Unnamed: 0' in df.columns: df.drop('Unnamed: 0', axis=1, inplace=True)
if 'id' in df.columns: df.drop('id', axis=1, inplace=True)

# Handle Missing Values (Imputation)
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0]) # Mode for text
        else:
            df[col] = df[col].fillna(df[col].mean())  # Mean for numerical

# Define Features (X) and Target (Y)
X = df.drop('Price', axis=1) 
Y = df['Price']

# Encode Categorical Features (X) to Integers
cat_cols = X.select_dtypes(['object']).columns
X[cat_cols] = X[cat_cols].apply(lambda x: pd.factorize(x)[0])

# Split Data (70% Train, 30% Test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print("Data Preparation and Splitting Complete.")

# --- 2. APPLY REGRESSION MODELS ---
print("\n--- 2. REGRESSION MODELS & EVALUATION ---")

# Dictionary of Regression Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Reg.': RandomForestRegressor(n_estimators=100, random_state=42),
    'Decision Tree Reg.': DecisionTreeRegressor(random_state=42),
    'K-Neighbors Reg.': KNeighborsRegressor(n_neighbors=5)
}

# Containers for Results
results = {'Names': [], 'R2 Score': [], 'MSE': []}

print("Regresser          R-Squared      MSE")
print("----------------------------------------")

for name, model in models.items():
    # Training and Prediction
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    r2 = metrics.r2_score(Y_test, y_pred)
    mse = metrics.mean_squared_error(Y_test, y_pred)
    
    # Store Results
    results['Names'].append(name)
    results['R2 Score'].append(r2)
    results['MSE'].append(mse)
    
    # Simple Print Results
    print(f"{name:<20} {r2:<14.4f} {mse:,.0f}")

# --- 3. VISUALIZATION ---
print("\n--- 3. VISUALIZATION ---")

# Graph 1: R-Squared Score Comparison
plt.figure(figsize=(10, 6))
plt.bar(results['Names'], results['R2 Score'], color=['#1f77b4', "#ff930e", '#2ca02c', '#d62728'], alpha=0.8, edgecolor='black')
plt.title("R-Squared Comparison (Laptop Price Prediction)")
plt.xlabel("Regression Algorithm")
plt.ylabel("R-Squared Score (Closer to 1 is Better)")
plt.ylim(0, 1.0) 
for i, v in enumerate(results['R2 Score']):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.show() 

# Graph 2: Mean Squared Error (MSE) Comparison
plt.figure(figsize=(10, 6))
plt.bar(results['Names'], results['MSE'], color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'], alpha=0.8, edgecolor='black')
plt.title("Mean Squared Error (MSE) Comparison")
plt.xlabel("Regression Algorithm")
plt.ylabel("MSE (Lower is Better)")
plt.yscale('log')
for i, v in enumerate(results['MSE']):
    plt.text(i, v * 1.05, f"{v:,.0f}", ha='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.show()

print("\nAll Lab 11 Tasks Completed.")