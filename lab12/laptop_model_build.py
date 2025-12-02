# ==========================================
# LAB 12: LAPTOP PRICE PREDICTION MODEL
# ==========================================
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
filename = 'laptop_data.csv'
try:
    df = pd.read_csv(filename, encoding='latin-1')
except:
    df = pd.read_csv(filename)

print("Data loaded...")

# 2. Data Cleaning
# Drop ID columns if they exist
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Clean RAM (Remove 'GB') and Weight (Remove 'kg')
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# 3. Separate Features (X) and Target (Y)
# We want to predict 'Price'
Y = df['Price']
X = df.drop('Price', axis=1)

# 4. Encode Categorical Data
# We use LabelEncoder to turn text (like "HP", "Intel Core i5") into numbers
encoders = {}
cat_columns = X.select_dtypes(['object']).columns

for col in cat_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# 5. Train Model
# Random Forest Regressor is excellent for price prediction
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Check accuracy
score = model.score(X_test, Y_test)
print(f"Model Trained. R2 Score (Accuracy): {score:.2f}")

# 6. Save Model and Encoders
data_to_save = {
    'model': model,
    'encoders': encoders,
    'feature_names': X.columns.tolist()
}

with open('laptop_price_model.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("Success! Model saved as 'laptop_price_model.pkl'")