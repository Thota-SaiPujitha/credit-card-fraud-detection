import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import streamlit as st
import time


def train_models(X_train, y_train, X_test, y_test, model_options,
                 handle_imbalance=True, imbalance_method='SMOTE',
                 hyperparameter_tuning=False):
    """
    Train multiple models for fraud detection.

    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training labels
    X_test : DataFrame
        Testing features
    y_test : Series
        Testing labels
    model_options : list
        List of model names to train
    handle_imbalance : bool
        Whether to handle class imbalance
    imbalance_method : str
        Method to handle imbalance ('SMOTE', 'RandomUnderSampler', 'Class Weights')
    hyperparameter_tuning : bool
        Whether to perform hyperparameter tuning

    Returns:
    --------
    models : dict
        Dictionary of trained models and their metrics
    best_model : estimator
        The best performing model
    best_model_name : str
        Name of the best model
    feature_importance : dict
        Feature importance for each model (if available)
    """
    models = {}
    best_f1 = 0
    best_model = None
    best_model_name = None
    feature_importance = {}

    # Handle class imbalance if requested
    if handle_imbalance:
        if imbalance_method == 'SMOTE':
            st.info("Applying SMOTE to handle class imbalance...")
            try:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                st.success(f"SMOTE applied. New class distribution: {np.bincount(y_resampled)}")
            except Exception as e:
                st.warning(f"Error applying SMOTE: {str(e)}. Using original data.")
                X_resampled, y_resampled = X_train, y_train
        elif imbalance_method == 'RandomUnderSampler':
            st.info("Applying Random Undersampling to handle class imbalance...")
            try:
                undersampler = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
                st.success(f"RandomUnderSampler applied. New class distribution: {np.bincount(y_resampled)}")
            except Exception as e:
                st.warning(f"Error applying RandomUnderSampler: {str(e)}. Using original data.")
                X_resampled, y_resampled = X_train, y_train
        elif imbalance_method == 'Class Weights':
            # Class weights will be applied during model training
            X_resampled, y_resampled = X_train, y_train
            st.info("Class weights will be applied during model training.")
        else:
            X_resampled, y_resampled = X_train, y_train
    else:
        X_resampled, y_resampled = X_train, y_train

    # Train Logistic Regression
    if "Logistic Regression" in model_options:
        st.info("Training Logistic Regression...")
        start_time = time.time()

        if hyperparameter_tuning:
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'solver': ['liblinear', 'saga']
            }

            if imbalance_method == 'Class Weights':
                grid = GridSearchCV(
                    LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )
            else:
                grid = GridSearchCV(
                    LogisticRegression(random_state=42, max_iter=1000),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )

            grid.fit(X_resampled, y_resampled)
            lr_model = grid.best_estimator_
            lr_params = grid.best_params_
            st.success(f"Best parameters: {lr_params}")
        else:
            if imbalance_method == 'Class Weights':
                lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
            else:
                lr_model = LogisticRegression(random_state=42, max_iter=1000)

            lr_model.fit(X_resampled, y_resampled)
            lr_params = lr_model.get_params()

        # Make predictions
        y_pred = lr_model.predict(X_test)
        y_pred_proba = lr_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Classification report and confusion matrix
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Training metrics
        y_train_pred = lr_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Store model and metrics
        models["Logistic Regression"] = {
            'model': lr_model,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc_roc': auc_roc,
                'classification_report': report,
                'confusion_matrix': cm,
                'train_accuracy': train_accuracy
            },
            'parameters': lr_params,
            'training_time': time.time() - start_time
        }

        # Check if this is the best model so far
        if f1 > best_f1:
            best_f1 = f1
            best_model = lr_model
            best_model_name = "Logistic Regression"

        # Get feature importance (coefficients)
        if hasattr(lr_model, 'coef_'):
            feature_importance["Logistic Regression"] = dict(zip(X_train.columns, lr_model.coef_[0]))

        st.success(f"Logistic Regression trained in {time.time() - start_time:.2f} seconds.")

    # Train Random Forest
    if "Random Forest" in model_options:
        st.info("Training Random Forest...")
        start_time = time.time()

        if hyperparameter_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            if imbalance_method == 'Class Weights':
                grid = GridSearchCV(
                    RandomForestClassifier(random_state=42, class_weight='balanced'),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )
            else:
                grid = GridSearchCV(
                    RandomForestClassifier(random_state=42),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )

            grid.fit(X_resampled, y_resampled)
            rf_model = grid.best_estimator_
            rf_params = grid.best_params_
            st.success(f"Best parameters: {rf_params}")
        else:
            if imbalance_method == 'Class Weights':
                rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            else:
                rf_model = RandomForestClassifier(random_state=42)

            rf_model.fit(X_resampled, y_resampled)
            rf_params = rf_model.get_params()

        # Make predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Classification report and confusion matrix
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Training metrics
        y_train_pred = rf_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Store model and metrics
        models["Random Forest"] = {
            'model': rf_model,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc_roc': auc_roc,
                'classification_report': report,
                'confusion_matrix': cm,
                'train_accuracy': train_accuracy
            },
            'parameters': rf_params,
            'training_time': time.time() - start_time
        }

        # Check if this is the best model so far
        if f1 > best_f1:
            best_f1 = f1
            best_model = rf_model
            best_model_name = "Random Forest"

        # Get feature importance
        if hasattr(rf_model, 'feature_importances_'):
            feature_importance["Random Forest"] = dict(zip(X_train.columns, rf_model.feature_importances_))

        st.success(f"Random Forest trained in {time.time() - start_time:.2f} seconds.")

    # Train XGBoost
    if "XGBoost" in model_options:
        st.info("Training XGBoost...")
        start_time = time.time()

        if hyperparameter_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }

            if imbalance_method == 'Class Weights':
                # Define sample weights
                fraud_ratio = sum(y_resampled) / len(y_resampled)
                scale_pos_weight = (1 - fraud_ratio) / fraud_ratio

                grid = GridSearchCV(
                    xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )
            else:
                grid = GridSearchCV(
                    xgb.XGBClassifier(random_state=42),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )

            grid.fit(X_resampled, y_resampled)
            xgb_model = grid.best_estimator_
            xgb_params = grid.best_params_
            st.success(f"Best parameters: {xgb_params}")
        else:
            if imbalance_method == 'Class Weights':
                # Define sample weights
                fraud_ratio = sum(y_resampled) / len(y_resampled)
                scale_pos_weight = (1 - fraud_ratio) / fraud_ratio

                xgb_model = xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
            else:
                xgb_model = xgb.XGBClassifier(random_state=42)

            xgb_model.fit(X_resampled, y_resampled)
            xgb_params = xgb_model.get_params()

        # Make predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Classification report and confusion matrix
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Training metrics
        y_train_pred = xgb_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Store model and metrics
        models["XGBoost"] = {
            'model': xgb_model,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc_roc': auc_roc,
                'classification_report': report,
                'confusion_matrix': cm,
                'train_accuracy': train_accuracy
            },
            'parameters': xgb_params,
            'training_time': time.time() - start_time
        }

        # Check if this is the best model so far
        if f1 > best_f1:
            best_f1 = f1
            best_model = xgb_model
            best_model_name = "XGBoost"

        # Get feature importance
        if hasattr(xgb_model, 'feature_importances_'):
            feature_importance["XGBoost"] = dict(zip(X_train.columns, xgb_model.feature_importances_))

        st.success(f"XGBoost trained in {time.time() - start_time:.2f} seconds.")

    # Train Decision Tree
    if "Decision Tree" in model_options:
        st.info("Training Decision Tree...")
        start_time = time.time()

        if hyperparameter_tuning:
            param_grid = {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }

            if imbalance_method == 'Class Weights':
                grid = GridSearchCV(
                    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )
            else:
                grid = GridSearchCV(
                    DecisionTreeClassifier(random_state=42),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )

            grid.fit(X_resampled, y_resampled)
            dt_model = grid.best_estimator_
            dt_params = grid.best_params_
            st.success(f"Best parameters: {dt_params}")
        else:
            if imbalance_method == 'Class Weights':
                dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
            else:
                dt_model = DecisionTreeClassifier(random_state=42)

            dt_model.fit(X_resampled, y_resampled)
            dt_params = dt_model.get_params()

        # Make predictions
        y_pred = dt_model.predict(X_test)
        y_pred_proba = dt_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Classification report and confusion matrix
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Training metrics
        y_train_pred = dt_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Store model and metrics
        models["Decision Tree"] = {
            'model': dt_model,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc_roc': auc_roc,
                'classification_report': report,
                'confusion_matrix': cm,
                'train_accuracy': train_accuracy
            },
            'parameters': dt_params,
            'training_time': time.time() - start_time
        }

        # Check if this is the best model so far
        if f1 > best_f1:
            best_f1 = f1
            best_model = dt_model
            best_model_name = "Decision Tree"

        # Get feature importance
        if hasattr(dt_model, 'feature_importances_'):
            feature_importance["Decision Tree"] = dict(zip(X_train.columns, dt_model.feature_importances_))

        st.success(f"Decision Tree trained in {time.time() - start_time:.2f} seconds.")

    # Train Support Vector Machine
    if "Support Vector Machine" in model_options:
        st.info("Training Support Vector Machine...")
        start_time = time.time()

        if hyperparameter_tuning:
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1]
            }

            if imbalance_method == 'Class Weights':
                grid = GridSearchCV(
                    SVC(random_state=42, class_weight='balanced', probability=True),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )
            else:
                grid = GridSearchCV(
                    SVC(random_state=42, probability=True),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )

            grid.fit(X_resampled, y_resampled)
            svm_model = grid.best_estimator_
            svm_params = grid.best_params_
            st.success(f"Best parameters: {svm_params}")
        else:
            if imbalance_method == 'Class Weights':
                svm_model = SVC(random_state=42, class_weight='balanced', probability=True)
            else:
                svm_model = SVC(random_state=42, probability=True)

            svm_model.fit(X_resampled, y_resampled)
            svm_params = svm_model.get_params()

        # Make predictions
        y_pred = svm_model.predict(X_test)
        y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Classification report and confusion matrix
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Training metrics
        y_train_pred = svm_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Store model and metrics
        models["Support Vector Machine"] = {
            'model': svm_model,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc_roc': auc_roc,
                'classification_report': report,
                'confusion_matrix': cm,
                'train_accuracy': train_accuracy
            },
            'parameters': svm_params,
            'training_time': time.time() - start_time
        }

        # Check if this is the best model so far
        if f1 > best_f1:
            best_f1 = f1
            best_model = svm_model
            best_model_name = "Support Vector Machine"

        # No direct feature importance for SVM

        st.success(f"Support Vector Machine trained in {time.time() - start_time:.2f} seconds.")

    return models, best_model, best_model_name, feature_importance


def evaluate_model(model, X_test, y_test, X_train, y_train):
    """
    Evaluate a trained model on test data.

    Parameters:
    -----------
    model : estimator
        The trained model
    X_test : DataFrame
        Testing features
    y_test : Series
        Testing labels
    X_train : DataFrame
        Training features
    y_train : Series
        Training labels

    Returns:
    --------
    results : dict
        Dictionary with evaluation results
    """
    results = {}

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Training predictions (for checking overfitting)
    y_train_pred = model.predict(X_train)

    # Calculate metrics
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['recall'] = recall_score(y_test, y_pred)
    results['f1'] = f1_score(y_test, y_pred)
    results['auc_roc'] = roc_auc_score(y_test, y_pred_proba)

    # Training metrics
    results['train_accuracy'] = accuracy_score(y_train, y_train_pred)

    # Calculate overfitting metrics
    results['overfit_accuracy'] = results['train_accuracy'] - results['accuracy']

    # Classification report and confusion matrix
    results['classification_report'] = classification_report(y_test, y_pred)
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)

    # Confusion matrix values
    tn, fp, fn, tp = results['confusion_matrix'].ravel()
    results['true_negatives'] = tn
    results['false_positives'] = fp
    results['false_negatives'] = fn
    results['true_positives'] = tp

    # Additional metrics
    results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    results['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    results['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    return results
