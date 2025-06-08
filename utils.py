import pandas as pd
import numpy as np
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import streamlit as st

def load_example_data():
    """
    Load example credit card fraud data
    """
    np.random.seed(42)
    n_samples = 10000
    n_fraud = 200

    time = np.random.uniform(0, 172800, n_samples)
    amount = np.random.exponential(scale=100, size=n_samples)

    normal_samples = n_samples - n_fraud
    v_normal = np.random.normal(loc=0, scale=1, size=(normal_samples, 10))
    v_fraud = np.random.normal(loc=-3, scale=2, size=(n_fraud, 10))
    V = np.vstack([v_normal, v_fraud])

    Class = np.zeros(n_samples)
    Class[normal_samples:] = 1

    idx = np.random.permutation(n_samples)
    data = pd.DataFrame({
        'Time': time[idx],
        'Amount': amount[idx],
        'Class': Class[idx]
    })

    for i in range(10):
        data[f'V{i + 1}'] = V[idx, i]

    return data

def preprocess_data(data, test_size=0.2, scaling_method='StandardScaler',
                    outlier_treatment=False, outlier_threshold=3.0,
                    apply_pca=False, pca_components=None,
                    feature_selection=False, n_features=None,
                    random_state=42):
    """
    Preprocess data for fraud detection
    """
    preprocessing_info = {}

    if 'Class' in data.columns:
        X = data.drop('Class', axis=1)
        y = data['Class']
    else:
        X = data.copy()
        y = None

    # Force numeric conversion and handle categorical data
    for column in X.columns:
        if X[column].dtype == 'object':
            # Replace any non-numeric values with NaN
            X[column] = pd.to_numeric(X[column], errors='coerce')
            # Fill NaN with -1
            X[column] = X[column].fillna(-1)
        # Ensure all columns are float64
        X[column] = X[column].astype('float64')

    # Handle any remaining missing values
    X = X.fillna(-1)


    # Scale features
    if scaling_method != 'None':
        if scaling_method == 'StandardScaler':
            scaler = StandardScaler()
        elif scaling_method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaling_method == 'RobustScaler':
            scaler = RobustScaler()

        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Split data if we have labels
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, None, y, None

    return X, y, X_train, X_test, y_train, y_test, preprocessing_info

def compute_statistics(data):
    """
    Compute fraud detection statistics
    """
    stats = {}

    if 'Class' not in data.columns or 'Amount' not in data.columns:
        return stats

    normal_data = data[data['Class'] == 0]
    fraud_data = data[data['Class'] == 1]

    stats['normal_count'] = len(normal_data)
    stats['fraud_count'] = len(fraud_data)
    stats['total_count'] = len(data)
    stats['fraud_percentage'] = (stats['fraud_count'] / stats['total_count']) * 100

    stats['normal_mean_amount'] = normal_data['Amount'].mean()
    stats['normal_median_amount'] = normal_data['Amount'].median()
    stats['normal_min_amount'] = normal_data['Amount'].min()
    stats['normal_max_amount'] = normal_data['Amount'].max()
    stats['normal_std_amount'] = normal_data['Amount'].std()

    stats['fraud_mean_amount'] = fraud_data['Amount'].mean()
    stats['fraud_median_amount'] = fraud_data['Amount'].median()
    stats['fraud_min_amount'] = fraud_data['Amount'].min()
    stats['fraud_max_amount'] = fraud_data['Amount'].max()
    stats['fraud_std_amount'] = fraud_data['Amount'].std()

    return stats