import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE

def clean_and_smote():
    # File paths
    base_dir = "DATASET"
    input_file = os.path.join(base_dir, "RELI Historical Data.csv")
    cleaned_file = os.path.join(base_dir, "RELI_Cleaned.csv")
    smote_file = os.path.join(base_dir, "RELI_Pure_SMOTE.csv")
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Initial Data Shape:", df.shape)
    
    # 1. Clean Numerical Columns
    cols_to_clean = ['Price', 'Open', 'High', 'Low']
    for col in cols_to_clean:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
    # Clean Volume
    def convert_volume(val):
        val = str(val).upper()
        if 'M' in val:
            return float(val.replace('M', '')) * 1000000
        elif 'K' in val:
            return float(val.replace('K', '')) * 1000
        elif 'B' in val:
            return float(val.replace('B', '')) * 1000000000
        else:
            try:
                return float(val)
            except:
                return 0.0
                
    df['Vol.'] = df['Vol.'].apply(convert_volume)
    
    # Clean Change %
    df['Change %'] = df['Change %'].astype(str).str.replace('%', '').astype(float)
    
    # 2. Parse Date and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
    # If some dates failed, try generic parser
    if df['Date'].isna().sum() > 0:
        df['Date'] = df['Date'].fillna(pd.to_datetime(df['Date'], errors='coerce'))
        
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    
    # Drop rows with missing dates or missing prices
    df = df.dropna(subset=['Date', 'Price'])
    
    # 3. Create Target Variable
    # 1 if Tomorrow's Price > Today's Price, else 0
    df['Tomorrow_Price'] = df['Price'].shift(-1)
    df['Target'] = (df['Tomorrow_Price'] > df['Price']).astype(int)
    
    # The last row won't have a 'Tomorrow_Price', so drop it
    df = df.dropna(subset=['Tomorrow_Price']).copy()
    df.drop(columns=['Tomorrow_Price'], inplace=True)
    
    print("Cleaned Data Shape:", df.shape)
    print("Class Distribution before SMOTE:\n", df['Target'].value_counts())
    
    # Save cleaned dataset
    df.to_csv(cleaned_file, index=False)
    print(f"Cleaned dataset saved to {cleaned_file}")
    
    # 4. Apply SMOTE
    # We drop 'Date' because SMOTE only works on numerical features.
    X = df.drop(columns=['Date', 'Target'])
    y = df['Target']
    
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)
    
    # Create final pure DataFrame
    df_pure = pd.DataFrame(X_sm, columns=X.columns)
    df_pure['Target'] = y_sm
    
    print("Pure Data Shape after SMOTE:", df_pure.shape)
    print("Class Distribution after SMOTE:\n", df_pure['Target'].value_counts())
    
    # Save SMOTE dataset
    df_pure.to_csv(smote_file, index=False)
    print(f"Pure SMOTE dataset saved to {smote_file}")

if __name__ == "__main__":
    clean_and_smote()
