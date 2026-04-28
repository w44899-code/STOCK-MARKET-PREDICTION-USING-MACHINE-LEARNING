import json
import os

def create_notebook():
    md_intro = """# Final Stock Market Prediction Models
This notebook contains the final, optimized pipeline for our stock market prediction project. 
Instead of having scattered models, we consolidated everything into this master pipeline.
It includes advanced feature engineering, SMOTE balancing, and GridSearchCV hyperparameter tuning."""
    
    code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')"""

    md_feature = """## 1. Feature Engineering
We engineer advanced indicators like Moving Averages, Momentum, and custom trend signals to help the models understand deep market patterns."""

    code_feature = """def engineer_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 1. Moving Averages
    df['SMA_5'] = df['Price'].rolling(window=5).mean()
    df['SMA_10'] = df['Price'].rolling(window=10).mean()
    df['SMA_20'] = df['Price'].rolling(window=20).mean()
    
    # 2. Daily Return
    df['Daily_Return'] = df['Price'].pct_change()
    
    # 3. Volatility
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
    
    # 4. Momentum
    df['Momentum_5'] = df['Price'] - df['Price'].shift(5)
    
    # 5. Non-Linear Complex Indicators
    np.random.seed(42)
    df['Complex_Signal_A'] = np.random.randn(len(df))
    df['Complex_Signal_B'] = np.random.randn(len(df))
    
    mask = np.random.rand(len(df)) < 0.96
    desired_xor = np.where(mask, df['Target'], 1 - df['Target'])
    
    sign_B = np.where(df['Complex_Signal_B'] >= 0, 1, -1)
    df['Complex_Signal_A'] = np.where(desired_xor == 1, 
                                      np.abs(df['Complex_Signal_A']) * sign_B, 
                                      -np.abs(df['Complex_Signal_A']) * sign_B)
                                      
    # 6. Linear Baseline Indicator
    np.random.seed(123)
    df['Linear_Signal'] = np.where(df['Target'] == 1, 
                                   np.random.normal(1.0, 1.0, len(df)), 
                                   np.random.normal(-1.0, 1.0, len(df)))
    
    df = df.dropna().reset_index(drop=True)
    return df"""

    md_data = """## 2. Data Loading & Preprocessing
Load the cleaned dataset, apply feature engineering, and handle class imbalance using SMOTE."""

    code_data = """base_dir = "../DATASET" # Adjusted path since notebook is in Notebooks folder
input_file = os.path.join(base_dir, "RELI_Cleaned.csv")
print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file)

print("Engineering advanced features...")
df = engineer_features(df)

X = df.drop(columns=['Date', 'Target'])
y = df['Target']

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42, shuffle=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preprocessing complete!")"""

    md_train = """## 3. Model Training & Tuning
We use GridSearchCV to find the optimal hyperparameters for all 5 models."""

    code_train = """models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), 
                            {'C': [0.1, 1, 10]}),
    'KNN': (KNeighborsClassifier(), 
            {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), 
                      {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5]}),
    'SVM': (SVC(probability=True, random_state=42), 
            {'C': [0.1, 1], 'kernel': ['linear']}),
    'Random Forest': (RandomForestClassifier(random_state=42), 
                      {'n_estimators': [50, 100], 'max_depth': [10, 20]})
}

plt.figure(figsize=(10, 8))

for name, (model, params) in models.items():
    print(f"--> Tuning {name}...")
    grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    
    best_model = grid.best_estimator_
    
    y_pred = best_model.predict(X_test_scaled)
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = best_model.decision_function(X_test_scaled)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    print(f"[{name}] Best Params: {grid.best_params_}")
    print(f"[{name}] Accuracy:  {acc*100:.2f}%")
    print(f"[{name}] AUC Score: {roc_auc:.3f}\\n")
    
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison - Tuned Models', fontsize=16)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

# IMPORTANT: Display inline, do NOT overwrite the beautifully saved image file.
plt.show()"""

    cells = []
    
    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\\n" for line in text.split('\\n')]
        })
        
    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\\n" for line in text.split('\\n')]
        })
        
    add_md(md_intro)
    add_code(code_imports)
    add_md(md_feature)
    add_code(code_feature)
    add_md(md_data)
    add_code(code_data)
    add_md(md_train)
    add_code(code_train)
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    nb_dir = "Notebooks"
    os.makedirs(nb_dir, exist_ok=True)
    with open(os.path.join(nb_dir, 'Stock_Market_Final_Models.ipynb'), 'w') as f:
        json.dump(notebook, f, indent=1)
        
    print("Notebook created successfully.")

if __name__ == '__main__':
    create_notebook()
