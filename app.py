import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import io
from model import train_models, evaluate_model
from visualization import (
    plot_feature_importance,
    plot_correlation_matrix,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)
from utils import preprocess_data, load_example_data, compute_statistics

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# App title and description
st.title("Credit Card Fraud Detection System")
st.markdown("""
This application helps detect fraudulent credit card transactions using machine learning.
Upload your transaction data or use the sample dataset to visualize patterns and detect fraud.
""")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio(
        "Choose the app mode",
        ["Home", "Data Exploration", "Model Training & Evaluation", "Prediction"]
    )

    st.header("Data Input")
    data_option = st.radio(
        "Choose data source",
        ["Use Example Data", "Upload Your Own Data"]
    )

    uploaded_file = None
    if data_option == "Upload Your Own Data":
        uploaded_file = st.file_uploader(
            "Upload CSV file with credit card transactions",
            type=["csv"]
        )

        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            st.info("""
            Your data should include transaction features and a 'Class' column with:
            - 0 for legitimate transactions
            - 1 for fraudulent transactions
            """)

    st.header("About")
    st.info("""
    This application uses machine learning to identify potentially fraudulent credit card
    transactions. It provides visualizations and metrics to help understand patterns
    in the data and evaluate model performance.
    """)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'preprocessing_steps' not in st.session_state:
    st.session_state.preprocessing_steps = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Load data
try:
    if st.session_state.data is None:
        if data_option == "Use Example Data":
            with st.spinner("Loading example data..."):
                data = load_example_data()
                st.session_state.data = data
        elif uploaded_file is not None:
            with st.spinner("Processing uploaded data..."):
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
        else:
            data = None
    else:
        data = st.session_state.data
except Exception as e:
    st.error(f"Error loading data: {e}")
    data = None

# Home page
if app_mode == "Home":
    st.header("Welcome to the Credit Card Fraud Detection System")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Purpose")
        st.write("""
        Credit card fraud is a significant challenge for financial institutions and users alike.
        This application helps identify potential fraud by:
        - Analyzing transaction patterns
        - Applying machine learning models
        - Visualizing suspicious activities
        - Providing real-time detection
        """)

        st.subheader("Getting Started")
        st.write("""
        1. Select a data source (example or upload your own)
        2. Explore the data through visualizations
        3. Train and evaluate fraud detection models
        4. Make predictions on new transactions
        """)

    with col2:
        st.subheader("Key Features")
        feature_list = [
            "Interactive data exploration",
            "Multiple machine learning models",
            "Performance metrics and comparisons",
            "Feature importance analysis",
            "Visual pattern recognition",
            "Real-time prediction capability"
        ]
        for feature in feature_list:
            st.markdown(f"✅ {feature}")

        st.subheader("Understanding Results")
        st.write("""
        - **Precision**: How many identified as fraud were actually fraudulent
        - **Recall**: What percentage of actual fraud was caught
        - **F1 Score**: Balance between precision and recall
        - **ROC-AUC**: Overall model performance measure
        """)

    st.info("To get started, select a navigation option from the sidebar to the left!")

# Data Exploration page
elif app_mode == "Data Exploration" and data is not None:
    st.header("Data Exploration")

    # Display basic info about the dataset
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", f"{len(data):,}")

    if 'Class' in data.columns:
        with col2:
            fraud_count = data['Class'].sum()
            st.metric("Fraudulent Transactions", f"{fraud_count:,}")

        with col3:
            fraud_percentage = (fraud_count / len(data)) * 100
            st.metric("Fraud Percentage", f"{fraud_percentage:.3f}%")

    # Show data sample
    with st.expander("View Data Sample"):
        st.dataframe(data.head(10))

    # Data statistics
    with st.expander("Data Statistics"):
        st.dataframe(data.describe())

    # Data visualizations
    st.subheader("Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Class Distribution",
        "Feature Correlation",
        "Feature Distribution",
        "Transaction Amount Analysis"
    ])

    with tab1:
        if 'Class' in data.columns:
            fig = plot_class_distribution(data)
            st.pyplot(fig)
        else:
            st.warning("No 'Class' column found in the data")

    with tab2:
        # Cap the correlation matrix at first 15 columns for visibility
        display_cols = min(15, len(data.columns))
        fig = plot_correlation_matrix(data.iloc[:, :display_cols])
        st.pyplot(fig)

        if display_cols < len(data.columns):
            st.info(f"Showing correlation for first {display_cols} columns for better visibility.")

    with tab3:
        # Feature distribution comparison
        st.subheader("Feature Distributions (Fraud vs. Normal)")

        if 'Class' in data.columns:
            # Select a feature to visualize
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_feature = st.selectbox("Select feature to visualize:", numeric_columns)

            # Create distribution plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Filter data for normal and fraudulent transactions
            normal_data = data[data['Class'] == 0][selected_feature]
            fraud_data = data[data['Class'] == 1][selected_feature]

            # Plot distributions
            sns.histplot(normal_data, ax=ax, label='Normal', alpha=0.5, kde=True)
            sns.histplot(fraud_data, ax=ax, label='Fraud', alpha=0.5, kde=True)

            plt.title(f'Distribution of {selected_feature}')
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("No 'Class' column found in the data")

    with tab4:
        # Time series analysis if time feature exists
        if 'Time' in data.columns and 'Amount' in data.columns and 'Class' in data.columns:
            st.subheader("Transaction Amount Over Time")

            fig, ax = plt.subplots(figsize=(12, 6))

            # Group by time intervals
            time_interval = st.slider("Time interval (hours)", 1, 24, 4)
            data['Time_Hour'] = data['Time'] / 3600  # Convert to hours
            data['Time_Group'] = (data['Time_Hour'] // time_interval).astype(int) * time_interval

            # Calculate average amount per time group and class
            time_amount = data.groupby(['Time_Group', 'Class'])['Amount'].mean().unstack().fillna(0)

            # Plot
            if 0 in time_amount.columns:
                ax.plot(time_amount.index, time_amount[0], label='Normal', color='blue')
            if 1 in time_amount.columns:
                ax.plot(time_amount.index, time_amount[1], label='Fraud', color='red')

            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Average Transaction Amount')
            ax.legend()
            st.pyplot(fig)
        elif 'Amount' in data.columns and 'Class' in data.columns:
            # If no time feature, show amount distribution
            st.subheader("Transaction Amount Distribution")

            fig, ax = plt.subplots(figsize=(10, 6))

            # Create boxplot for amount by class
            sns.boxplot(x='Class', y='Amount', data=data, ax=ax)
            ax.set_title('Transaction Amount by Class')
            ax.set_xlabel('Class (0=Normal, 1=Fraud)')
            ax.set_ylabel('Amount')

            st.pyplot(fig)

            # Add log scale option for better visibility
            if st.checkbox("Use Log Scale for Amount"):
                fig, ax = plt.subplots(figsize=(10, 6))
                data_copy = data.copy()
                data_copy['Log_Amount'] = np.log1p(data_copy['Amount'])

                sns.boxplot(x='Class', y='Log_Amount', data=data_copy, ax=ax)
                ax.set_title('Log Transaction Amount by Class')
                ax.set_xlabel('Class (0=Normal, 1=Fraud)')
                ax.set_ylabel('Log(Amount + 1)')

                st.pyplot(fig)
        else:
            st.warning("Missing 'Time', 'Amount', or 'Class' columns for transaction analysis")

    # Additional statistics about the data
    if 'Class' in data.columns:
        st.subheader("Statistical Insights")
        stats = compute_statistics(data)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Transaction Statistics:**")
            stats_df = pd.DataFrame({
                'Metric': ['Mean Transaction Amount', 'Median Transaction Amount',
                           'Max Transaction Amount', 'Standard Deviation'],
                'Normal': [
                    f"${stats['normal_mean_amount']:.2f}",
                    f"${stats['normal_median_amount']:.2f}",
                    f"${stats['normal_max_amount']:.2f}",
                    f"${stats['normal_std_amount']:.2f}"
                ],
                'Fraud': [
                    f"${stats['fraud_mean_amount']:.2f}",
                    f"${stats['fraud_median_amount']:.2f}",
                    f"${stats['fraud_max_amount']:.2f}",
                    f"${stats['fraud_std_amount']:.2f}"
                ]
            })
            st.table(stats_df)

        with col2:
            # Create a radar chart for feature comparison
            if len(data.columns) > 2:  # Need at least a few features
                feature_stats = {}

                # Take a few numeric features (excluding Class)
                numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if 'Class' in numeric_cols:
                    numeric_cols.remove('Class')

                # Limit to first 5 features for visibility
                display_features = numeric_cols[:5]

                for feature in display_features:
                    norm_mean = data[data['Class'] == 0][feature].mean()
                    fraud_mean = data[data['Class'] == 1][feature].mean()

                    # Normalize to a 0-1 scale for comparison
                    max_val = max(norm_mean, fraud_mean)
                    if max_val != 0:  # Avoid division by zero
                        feature_stats[feature] = {
                            'Normal': norm_mean / max_val,
                            'Fraud': fraud_mean / max_val
                        }

                if feature_stats:
                    # Create a radar chart
                    categories = list(feature_stats.keys())

                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, polar=True)

                    # Number of variables
                    N = len(categories)

                    # What will be the angle of each axis in the plot
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # Close the loop

                    # Draw one axe per variable and add labels
                    plt.xticks(angles[:-1], categories)

                    # Draw the normal transaction data
                    normal_values = [feature_stats[cat]['Normal'] for cat in categories]
                    normal_values += normal_values[:1]  # Close the loop
                    ax.plot(angles, normal_values, linewidth=1, linestyle='solid', label='Normal')
                    ax.fill(angles, normal_values, alpha=0.1)

                    # Draw the fraud transaction data
                    fraud_values = [feature_stats[cat]['Fraud'] for cat in categories]
                    fraud_values += fraud_values[:1]  # Close the loop
                    ax.plot(angles, fraud_values, linewidth=1, linestyle='solid', label='Fraud')
                    ax.fill(angles, fraud_values, alpha=0.1)

                    # Add legend
                    plt.legend(loc='upper right')
                    plt.title('Feature Comparison (Normalized)')

                    st.pyplot(fig)
                    st.caption(
                        "Radar chart showing normalized feature comparison between normal and fraudulent transactions")
                else:
                    st.info("Not enough numeric features for comparison visualization")
            else:
                st.info("Not enough features for comparison visualization")

# Model Training & Evaluation page
elif app_mode == "Model Training & Evaluation" and data is not None:
    st.header("Model Training & Evaluation")

    # Check if the data contains the required 'Class' column
    if 'Class' not in data.columns:
        st.error("The data does not contain a 'Class' column for fraud labels. Please use appropriate data.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Training Settings")

            # Model selection
            model_options = st.multiselect(
                "Select models to train:",
                ["Logistic Regression", "Random Forest", "XGBoost", "Decision Tree", "Support Vector Machine"],
                default=["Logistic Regression", "Random Forest", "XGBoost"]
            )

            # Test set size
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100

            # Handling imbalanced data
            handle_imbalance = st.checkbox("Handle imbalanced data", value=True)
            imbalance_method = None
            if handle_imbalance:
                imbalance_method = st.selectbox(
                    "Imbalance handling method:",
                    ["SMOTE", "RandomUnderSampler", "Class Weights"]
                )

            # Feature selection option
            feature_selection = st.checkbox("Apply feature selection", value=False)
            n_features = None
            if feature_selection:
                # Determine max number of features (excluding 'Class')
                max_features = len(data.columns) - 1
                n_features = st.slider("Number of features to select", 1, max_features, max_features // 2)

            # Hyperparameter tuning option
            hyperparameter_tuning = st.checkbox("Apply hyperparameter tuning", value=False)

        with col2:
            st.subheader("Feature Engineering")

            # Data scaling
            scaling_method = st.selectbox(
                "Scaling method:",
                ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"]
            )

            # Outlier treatment
            outlier_treatment = st.checkbox("Remove outliers", value=False)
            outlier_threshold = None
            if outlier_treatment:
                outlier_threshold = st.slider("Outlier threshold (Z-score)", 2.0, 5.0, 3.0, 0.1)

            # PCA option
            apply_pca = st.checkbox("Apply PCA for dimensionality reduction", value=False)
            pca_components = None
            if apply_pca:
                # Determine max number of components (excluding 'Class')
                max_components = len(data.columns) - 1
                pca_components = st.slider("Number of PCA components", 1, max_components, max_components // 2)

        # Button to start training
        train_button = st.button("Train Models")

        if train_button or st.session_state.models is not None:
            if train_button:
                if not model_options:
                    st.error("Please select at least one model to train.")
                else:
                    # Store preprocessing settings in session state
                    st.session_state.preprocessing_steps = {
                        'scaling_method': scaling_method,
                        'outlier_treatment': outlier_treatment,
                        'outlier_threshold': outlier_threshold,
                        'apply_pca': apply_pca,
                        'pca_components': pca_components,
                        'feature_selection': feature_selection,
                        'n_features': n_features,
                        'handle_imbalance': handle_imbalance,
                        'imbalance_method': imbalance_method
                    }

                    with st.spinner("Preprocessing data..."):
                        try:
                            # Preprocess the data
                            X, y, X_train, X_test, y_train, y_test, preprocessing_info = preprocess_data(
                                data,
                                test_size=test_size,
                                scaling_method=scaling_method,
                                outlier_treatment=outlier_treatment,
                                outlier_threshold=outlier_threshold,
                                apply_pca=apply_pca,
                                pca_components=pca_components,
                                feature_selection=feature_selection,
                                n_features=n_features
                            )

                            # Store the preprocessed data in session state
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test

                            st.success("Data preprocessing completed.")

                            # Training the selected models
                            with st.spinner("Training models... This may take a few minutes."):
                                models, best_model, best_model_name, feature_importance = train_models(
                                    X_train,
                                    y_train,
                                    X_test,
                                    y_test,
                                    model_options,
                                    handle_imbalance=handle_imbalance,
                                    imbalance_method=imbalance_method,
                                    hyperparameter_tuning=hyperparameter_tuning
                                )

                                # Store the trained models in session state
                                st.session_state.models = models
                                st.session_state.best_model = best_model
                                st.session_state.best_model_name = best_model_name
                                st.session_state.feature_importance = feature_importance

                                # Evaluate the best model
                                evaluation_results = evaluate_model(
                                    best_model,
                                    X_test,
                                    y_test,
                                    X_train,
                                    y_train
                                )

                                # Store evaluation results
                                st.session_state.evaluation_results = evaluation_results

                                st.success(f"Models trained successfully! Best model: {best_model_name}")
                        except Exception as e:
                            st.error(f"Error during training: {str(e)}")

            # If models have been trained, display results
            if st.session_state.models is not None:
                st.subheader("Model Performance Comparison")

                # Create a DataFrame to compare model performance
                model_names = list(st.session_state.models.keys())
                model_metrics = {name: st.session_state.models[name]['metrics'] for name in model_names}

                metrics_df = pd.DataFrame({
                    'Model': model_names,
                    'Accuracy': [model_metrics[name]['accuracy'] for name in model_names],
                    'Precision': [model_metrics[name]['precision'] for name in model_names],
                    'Recall': [model_metrics[name]['recall'] for name in model_names],
                    'F1 Score': [model_metrics[name]['f1'] for name in model_names],
                    'AUC-ROC': [model_metrics[name]['auc_roc'] for name in model_names]
                })

                # Sort by F1 score (typically a good balanced metric for fraud detection)
                metrics_df = metrics_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

                # Display metrics as a styled DataFrame
                st.dataframe(
                    metrics_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']))

                # Add a model selector for detailed analysis
                selected_model_name = st.selectbox(
                    "Select a model for detailed analysis:",
                    model_names,
                    index=model_names.index(
                        st.session_state.best_model_name) if st.session_state.best_model_name in model_names else 0,
                    key="analysis_model_selector"
                )

                st.session_state.selected_model = selected_model_name

                # Show detailed performance of the selected model
                st.subheader(f"Detailed Analysis: {selected_model_name}")

                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Confusion Matrix",
                    "ROC Curve",
                    "Precision-Recall Curve",
                    "Feature Importance",
                    "Model Details"
                ])

                with tab1:
                    # Confusion Matrix
                    selected_model_obj = st.session_state.models[selected_model_name]['model']
                    conf_matrix_fig = plot_confusion_matrix(
                        selected_model_obj,
                        st.session_state.X_test,
                        st.session_state.y_test
                    )
                    st.pyplot(conf_matrix_fig)

                    # Add textual explanation
                    st.write("""
                    **Confusion Matrix Explanation:**
                    - **True Negative (TN)**: Correctly identified legitimate transactions
                    - **False Positive (FP)**: Legitimate transactions incorrectly flagged as fraud
                    - **False Negative (FN)**: Fraudulent transactions missed by the model
                    - **True Positive (TP)**: Correctly identified fraudulent transactions
                    """)

                with tab2:
                    # ROC Curve
                    roc_curve_fig = plot_roc_curve(
                        selected_model_obj,
                        st.session_state.X_test,
                        st.session_state.y_test
                    )
                    st.pyplot(roc_curve_fig)

                    st.write("""
                    **ROC Curve Explanation:**
                    The ROC curve shows the trade-off between sensitivity (True Positive Rate) and specificity
                    (1 - False Positive Rate). The closer the curve is to the top-left corner, the better the model.
                    The Area Under the Curve (AUC) is a measure of the model's ability to distinguish between classes.
                    """)

                with tab3:
                    # Precision-Recall Curve
                    pr_curve_fig = plot_precision_recall_curve(
                        selected_model_obj,
                        st.session_state.X_test,
                        st.session_state.y_test
                    )
                    st.pyplot(pr_curve_fig)

                    st.write("""
                    **Precision-Recall Curve Explanation:**
                    For imbalanced classification like fraud detection, the Precision-Recall curve is often more
                    informative than the ROC curve. It focuses on the model's performance on the positive (fraud) class.
                    - **Precision**: How many selected items are relevant
                    - **Recall**: How many relevant items are selected
                    """)

                with tab4:
                    # Feature Importance
                    # Check if feature importance is available for the selected model
                    if selected_model_name in st.session_state.feature_importance:
                        feature_imp = st.session_state.feature_importance[selected_model_name]
                        feature_imp_fig = plot_feature_importance(feature_imp)
                        st.pyplot(feature_imp_fig)

                        st.write("""
                        **Feature Importance Explanation:**
                        This chart shows which features have the most influence on the model's predictions.
                        Higher values indicate that the feature has a stronger effect on determining whether
                        a transaction is fraudulent or legitimate.
                        """)
                    else:
                        st.info(f"Feature importance not available for {selected_model_name}")

                with tab5:
                    # Model Details
                    st.write("**Model Performance Metrics:**")
                    metrics = model_metrics[selected_model_name]

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    col2.metric("Precision", f"{metrics['precision']:.4f}")
                    col3.metric("Recall", f"{metrics['recall']:.4f}")
                    col4.metric("F1 Score", f"{metrics['f1']:.4f}")

                    st.markdown("---")

                    st.write("**Classification Report:**")
                    if 'classification_report' in metrics:
                        st.text(metrics['classification_report'])

                    st.markdown("---")

                    # Show confusion matrix values
                    if 'confusion_matrix' in metrics:
                        st.write("**Confusion Matrix Values:**")
                        cm = metrics['confusion_matrix']
                        cm_df = pd.DataFrame(
                            cm,
                            index=['Actual: Normal', 'Actual: Fraud'],
                            columns=['Predicted: Normal', 'Predicted: Fraud']
                        )
                        st.dataframe(cm_df)

                    st.markdown("---")

                    # Model parameters
                    if 'parameters' in st.session_state.models[selected_model_name]:
                        st.write("**Model Parameters:**")
                        params = st.session_state.models[selected_model_name]['parameters']
                        # Convert parameters to a more readable format
                        params_df = pd.DataFrame([params]).T.reset_index()
                        params_df.columns = ['Parameter', 'Value']
                        st.dataframe(params_df)

                # Additional performance visualization
                st.subheader("Performance Overview")

                # Create a radar chart comparing key metrics across models
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, polar=True)

                # The categories that we want to compare
                categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

                # Number of categories
                N = len(categories)

                # What will be the angle of each axis in the plot
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop

                # Add labels
                plt.xticks(angles[:-1], categories)

                # Select up to 5 top models by F1 score for comparison
                top_models = metrics_df.head(min(5, len(metrics_df)))

                # Plot each model
                for i, (idx, row) in enumerate(top_models.iterrows()):
                    values = row[categories].values.tolist()
                    values += values[:1]  # Close the loop

                    ax.plot(angles, values, linewidth=1, linestyle='solid', label=row['Model'])
                    ax.fill(angles, values, alpha=0.1)

                # Add legend
                plt.legend(loc='upper right')
                plt.title('Model Comparison')

                st.pyplot(fig)

# Prediction page
elif app_mode == "Prediction" and data is not None:
    st.header("Fraud Detection Prediction")

    # Check if models have been trained
    if st.session_state.models is None:
        st.warning("Please train models first before making predictions.")
        if st.button("Go to Model Training"):
            st.session_state.app_mode = "Model Training & Evaluation"
            st.rerun()
    else:
        st.subheader("Make Predictions")

        # Create tabs for different prediction modes
        tab1, tab2 = st.tabs(["Predict on Test Data", "Predict on New Data"])

        with tab1:
            # Predict on test data
            st.write("Use the trained model to predict fraud on the test dataset.")

            # Select model for prediction
            model_names = list(st.session_state.models.keys())
            selected_model_name = st.selectbox(
                "Select a model for prediction:",
                model_names,
                index=model_names.index(
                    st.session_state.best_model_name) if st.session_state.best_model_name in model_names else 0,
                key="prediction_model_selector"
            )

            if st.button("Run Prediction on Test Data"):
                with st.spinner("Making predictions..."):
                    try:
                        # Get the selected model
                        selected_model = st.session_state.models[selected_model_name]['model']

                        # Make predictions
                        y_pred = selected_model.predict(st.session_state.X_test)
                        y_pred_proba = selected_model.predict_proba(st.session_state.X_test)[:, 1]

                        # Create a DataFrame to display results
                        results_df = pd.DataFrame({
                            'Actual Class': st.session_state.y_test,
                            'Predicted Class': y_pred,
                            'Fraud Probability': y_pred_proba
                        })

                        # Add a column to highlight incorrect predictions
                        results_df['Correct Prediction'] = results_df['Actual Class'] == results_df['Predicted Class']

                        # Display results
                        st.subheader("Prediction Results (Sample)")
                        st.dataframe(results_df.head(20).style.apply(
                            lambda x: ['background-color: #ffcccc' if not x['Correct Prediction'] else '' for _ in x],
                            axis=1
                        ))

                        # Show summary statistics
                        st.subheader("Prediction Summary")

                        col1, col2, col3 = st.columns(3)

                        # Number of transactions predicted as fraud
                        pred_fraud_count = y_pred.sum()
                        col1.metric(
                            "Predicted Fraud Count",
                            f"{pred_fraud_count:,}",
                            f"{pred_fraud_count - st.session_state.y_test.sum():+,}"
                        )

                        # Accuracy on test set
                        accuracy = (results_df['Correct Prediction'].sum() / len(results_df))
                        col2.metric("Accuracy", f"{accuracy:.4f}")

                        # False positives
                        false_positives = (
                                    (results_df['Predicted Class'] == 1) & (results_df['Actual Class'] == 0)).sum()
                        col3.metric(
                            "False Positives",
                            f"{false_positives:,}",
                            f"{((false_positives / len(results_df)) * 100):.2f}%",
                            delta_color="inverse"
                        )

                        # Visualization of prediction distribution
                        st.subheader("Prediction Distribution")

                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Create histogram of fraud probabilities
                        sns.histplot(
                            data=results_df,
                            x='Fraud Probability',
                            hue='Actual Class',
                            bins=50,
                            alpha=0.7,
                            ax=ax
                        )

                        plt.title('Distribution of Fraud Probabilities')
                        plt.xlabel('Fraud Probability')
                        plt.ylabel('Frequency')
                        plt.legend(['Normal', 'Fraud'])

                        st.pyplot(fig)

                        # Interactive threshold selection
                        st.subheader("Adjust Prediction Threshold")
                        st.write("""
                        By default, transactions with a fraud probability >= 0.5 are classified as fraudulent.
                        You can adjust this threshold to balance between false positives and false negatives.
                        """)

                        threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.5, 0.01)

                        # Recalculate predictions with the new threshold
                        results_df['Predicted Class (Adjusted)'] = (
                                    results_df['Fraud Probability'] >= threshold).astype(int)
                        results_df['Correct Prediction (Adjusted)'] = results_df['Actual Class'] == results_df[
                            'Predicted Class (Adjusted)']

                        # Calculate metrics with the new threshold
                        tn, fp, fn, tp = confusion_matrix(
                            results_df['Actual Class'],
                            results_df['Predicted Class (Adjusted)']
                        ).ravel()

                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                        # Display the new metrics
                        col1, col2, col3, col4 = st.columns(4)

                        col1.metric("Precision", f"{precision:.4f}")
                        col2.metric("Recall", f"{recall:.4f}")
                        col3.metric("F1 Score", f"{f1:.4f}")
                        col4.metric("False Positives", f"{fp:,}")

                        # Show confusion matrix with the new threshold
                        st.subheader("Confusion Matrix (Adjusted Threshold)")

                        conf_matrix_fig = plt.figure(figsize=(8, 6))
                        cm = confusion_matrix(
                            results_df['Actual Class'],
                            results_df['Predicted Class (Adjusted)']
                        )
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt='d',
                            cmap='Blues',
                            xticklabels=['Normal', 'Fraud'],
                            yticklabels=['Normal', 'Fraud']
                        )
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        plt.title(f'Confusion Matrix (Threshold: {threshold:.2f})')

                        st.pyplot(conf_matrix_fig)

                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")

        with tab2:
            # Predict on new data
            st.write("Upload new transaction data for prediction or input a single transaction manually.")

            input_method = st.radio(
                "Input method:",
                ["Upload CSV", "Manual Input"]
            )

            if input_method == "Upload CSV":
                # File upload for new data
                new_data_file = st.file_uploader(
                    "Upload CSV file with new transactions",
                    type=["csv"]
                )

                if new_data_file is not None:
                    try:
                        # Load the new data
                        new_data = pd.read_csv(new_data_file)

                        # Display a sample of the new data
                        st.subheader("New Data Sample")
                        st.dataframe(new_data.head())

                        # Select model for prediction
                        model_names = list(st.session_state.models.keys())
                        selected_model_name = st.selectbox(
                            "Select a model for new data prediction:",
                            model_names,
                            index=model_names.index(
                                st.session_state.best_model_name) if st.session_state.best_model_name in model_names else 0,
                            key="new_data_model_selector"
                        )

                        # Button to run prediction
                        if st.button("Run Prediction on New Data"):
                            with st.spinner("Processing new data and making predictions..."):
                                # Check if 'Class' column exists in new data
                                has_class_column = 'Class' in new_data.columns

                                # Preprocess the new data
                                # We need to apply the same preprocessing steps as during training
                                if has_class_column:
                                    new_X = new_data.drop('Class', axis=1)
                                    new_y = new_data['Class']
                                else:
                                    new_X = new_data.copy()

                                # Apply preprocessing (simplified for demo)
                                # In a real application, you would apply the exact same preprocessing pipeline
                                if st.session_state.preprocessing_steps:
                                    scaling_method = st.session_state.preprocessing_steps['scaling_method']
                                    if scaling_method != "None":
                                        # For this demo, we'll just standardize the data
                                        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

                                        if scaling_method == "StandardScaler":
                                            scaler = StandardScaler()
                                        elif scaling_method == "MinMaxScaler":
                                            scaler = MinMaxScaler()
                                        elif scaling_method == "RobustScaler":
                                            scaler = RobustScaler()

                                        # Fit the scaler on the training data
                                        numeric_cols = new_X.select_dtypes(include=['float64', 'int64']).columns
                                        if len(numeric_cols) > 0:
                                            new_X[numeric_cols] = scaler.fit_transform(new_X[numeric_cols])

                                # Get the selected model
                                selected_model = st.session_state.models[selected_model_name]['model']

                                # Make predictions
                                new_pred = selected_model.predict(new_X)
                                new_pred_proba = selected_model.predict_proba(new_X)[:, 1]

                                # Create a DataFrame with the results
                                results_df = pd.DataFrame({
                                    'Predicted Class': new_pred,
                                    'Fraud Probability': new_pred_proba
                                })

                                if has_class_column:
                                    results_df['Actual Class'] = new_y
                                    results_df['Correct Prediction'] = results_df['Actual Class'] == results_df[
                                        'Predicted Class']

                                # Join results with the original data
                                all_results = pd.concat([new_data, results_df], axis=1)

                                # Display results
                                st.subheader("Prediction Results")
                                st.dataframe(all_results.head(50))

                                # Show summary of predictions
                                st.subheader("Prediction Summary")

                                col1, col2 = st.columns(2)

                                # Number of transactions predicted as fraud
                                fraud_count = new_pred.sum()
                                fraud_percentage = (fraud_count / len(new_pred)) * 100

                                col1.metric("Predicted Fraud Count", f"{fraud_count:,}")
                                col2.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")

                                # Download the results
                                csv = all_results.to_csv(index=False)
                                st.download_button(
                                    label="Download Prediction Results",
                                    data=csv,
                                    file_name="fraud_prediction_results.csv",
                                    mime="text/csv"
                                )

                                # Visualization of prediction distribution
                                st.subheader("Prediction Distribution")

                                fig, ax = plt.subplots(figsize=(10, 6))

                                # Create histogram of fraud probabilities
                                sns.histplot(
                                    data=results_df,
                                    x='Fraud Probability',
                                    bins=50,
                                    alpha=0.7,
                                    ax=ax
                                )

                                plt.title('Distribution of Fraud Probabilities')
                                plt.xlabel('Fraud Probability')
                                plt.ylabel('Frequency')

                                st.pyplot(fig)

                                # If we have the actual class, show more metrics
                                if has_class_column:
                                    # Confusion matrix
                                    st.subheader("Confusion Matrix")

                                    conf_matrix_fig = plt.figure(figsize=(8, 6))
                                    cm = confusion_matrix(new_y, new_pred)
                                    sns.heatmap(
                                        cm,
                                        annot=True,
                                        fmt='d',
                                        cmap='Blues',
                                        xticklabels=['Normal', 'Fraud'],
                                        yticklabels=['Normal', 'Fraud']
                                    )
                                    plt.xlabel('Predicted')
                                    plt.ylabel('Actual')
                                    plt.title('Confusion Matrix')

                                    st.pyplot(conf_matrix_fig)

                                    # Performance metrics
                                    # Metrics are now imported at the top of the file

                                    accuracy = accuracy_score(new_y, new_pred)
                                    precision = precision_score(new_y, new_pred)
                                    recall = recall_score(new_y, new_pred)
                                    f1 = f1_score(new_y, new_pred)

                                    col1, col2, col3, col4 = st.columns(4)

                                    col1.metric("Accuracy", f"{accuracy:.4f}")
                                    col2.metric("Precision", f"{precision:.4f}")
                                    col3.metric("Recall", f"{recall:.4f}")
                                    col4.metric("F1 Score", f"{f1:.4f}")

                    except Exception as e:
                        st.error(f"Error processing new data: {str(e)}")

            else:  # Manual Input
                st.write("""
                Enter transaction features manually. For a real credit card fraud detection system,
                you would need to know which features to input. For this demo, we'll use simplified features.
                """)

                # Create input fields for key features
                st.subheader("Transaction Features")

                # Create columns for input
                col1, col2 = st.columns(2)

                # Create some sample input fields
                # In a real application, you would need to match the exact features used during training
                with col1:
                    time = st.number_input("Time (seconds from first transaction)", 0, 100000, 50000)
                    amount = st.number_input("Transaction Amount", 0.01, 10000.0, 100.0)
                    v1 = st.number_input("V1 (PCA feature)", -10.0, 10.0, 0.0)
                    v2 = st.number_input("V2 (PCA feature)", -10.0, 10.0, 0.0)
                    v3 = st.number_input("V3 (PCA feature)", -10.0, 10.0, 0.0)
                    v4 = st.number_input("V4 (PCA feature)", -10.0, 10.0, 0.0)

                with col2:
                    v5 = st.number_input("V5 (PCA feature)", -10.0, 10.0, 0.0)
                    v6 = st.number_input("V6 (PCA feature)", -10.0, 10.0, 0.0)
                    v7 = st.number_input("V7 (PCA feature)", -10.0, 10.0, 0.0)
                    v8 = st.number_input("V8 (PCA feature)", -10.0, 10.0, 0.0)
                    v9 = st.number_input("V9 (PCA feature)", -10.0, 10.0, 0.0)
                    v10 = st.number_input("V10 (PCA feature)", -10.0, 10.0, 0.0)

                # Create a dataframe with the input values
                input_data = pd.DataFrame({
                    'Time': [time],
                    'V1': [v1],
                    'V2': [v2],
                    'V3': [v3],
                    'V4': [v4],
                    'V5': [v5],
                    'V6': [v6],
                    'V7': [v7],
                    'V8': [v8],
                    'V9': [v9],
                    'V10': [v10],
                    'Amount': [amount]
                })

                # Add missing columns from the training data if necessary
                # This is a simplified version for the demo
                # In a real application, you would need to ensure all features match

                # Select model for prediction
                model_names = list(st.session_state.models.keys())
                selected_model_name = st.selectbox(
                    "Select a model for prediction:",
                    model_names,
                    index=model_names.index(
                        st.session_state.best_model_name) if st.session_state.best_model_name in model_names else 0,
                    key="manual_input_model_selector"
                )

                # Button to run prediction
                if st.button("Predict"):
                    with st.spinner("Making prediction..."):
                        try:
                            # Get the selected model
                            selected_model = st.session_state.models[selected_model_name]['model']

                            # Make prediction
                            # Note: In a real application, you would need to preprocess the input data
                            # the same way as during training
                            pred = selected_model.predict(input_data)[0]
                            pred_proba = selected_model.predict_proba(input_data)[0, 1]

                            # Display the result
                            st.subheader("Prediction Result")

                            if pred == 1:
                                st.error(f"⚠️ Prediction: **FRAUDULENT TRANSACTION** (Probability: {pred_proba:.4f})")
                            else:
                                st.success(
                                    f"✅ Prediction: **LEGITIMATE TRANSACTION** (Fraud Probability: {pred_proba:.4f})")

                            # Explanation of the prediction
                            st.subheader("Prediction Explanation")

                            st.write(f"""
                            The model '{selected_model_name}' predicts that this transaction is 
                            {'fraudulent' if pred == 1 else 'legitimate'} with a fraud probability of {pred_proba:.4f}.
                            """)

                            # Show the threshold
                            st.write(f"""
                            The default threshold for classification is 0.5, meaning transactions with a fraud 
                            probability >= 0.5 are classified as fraudulent.
                            """)

                            # Show a gauge chart for the probability
                            gauge_fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})

                            # Define the colors for the gauge
                            cmap = plt.cm.RdYlGn_r
                            norm = plt.Normalize(0, 1)
                            colors = cmap(norm(np.linspace(0, 1, 100)))

                            # Draw the gauge
                            ax.set_theta_direction(-1)
                            ax.set_theta_offset(np.pi / 2.0)

                            # Add background arc
                            ax.bar(
                                x=np.linspace(0, np.pi, 100),
                                height=0.8,
                                width=np.pi / 100,
                                bottom=0.2,
                                color=colors,
                                edgecolor="gray",
                                linewidth=0.1,
                                alpha=0.6
                            )

                            # Add the value arc
                            ax.bar(
                                x=np.linspace(0, np.pi * pred_proba, 100),
                                height=0.8,
                                width=np.pi / 100,
                                bottom=0.2,
                                color=cmap(norm(pred_proba)),
                                edgecolor="black",
                                linewidth=0.1
                            )

                            # Remove unnecessary elements
                            ax.set_axis_off()

                            # Add text in the center
                            plt.text(
                                0, 0, f"{pred_proba:.2%}",
                                ha='center', va='center', fontsize=24
                            )

                            # Add labels
                            plt.text(-0.2, -0.15, "0", fontsize=14)
                            plt.text(np.pi + 0.2, -0.15, "1", fontsize=14)
                            plt.text(np.pi / 2, 1.2, "Fraud Probability", fontsize=16, ha='center')

                            st.pyplot(gauge_fig)

                            if hasattr(selected_model,
                                       'feature_importances_') and selected_model_name in st.session_state.feature_importance:
                                # Get feature importances for this specific prediction
                                st.subheader("Feature Contribution")
                                st.write("The chart below shows how each feature contributed to this prediction:")

                                # Get the feature importance
                                feature_imp = st.session_state.feature_importance[selected_model_name]

                                # Multiply input values by feature importance for a simplified explanation
                                # Note: This is a simplified approach and not proper SHAP values
                                feature_names = list(input_data.columns)
                                importance_values = []

                                for feature in feature_names:
                                    if feature in feature_imp:
                                        imp = feature_imp[feature] * input_data[feature].values[0]
                                        importance_values.append((feature, imp))

                                # Sort by absolute importance
                                importance_values.sort(key=lambda x: abs(x[1]), reverse=True)

                                # Create a horizontal bar chart
                                fig, ax = plt.subplots(figsize=(10, 6))

                                features = [x[0] for x in importance_values]
                                importances = [x[1] for x in importance_values]

                                # Choose colors based on importance values
                                colors = ['red' if x < 0 else 'green' for x in importances]

                                # Create the bar chart
                                ax.barh(features, importances, color=colors)

                                # Add a vertical line at x=0
                                ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)

                                plt.title('Feature Contribution to Prediction')
                                plt.xlabel('Contribution')

                                st.pyplot(fig)

                                st.write("""
                                Note: Positive values (green) push towards legitimate classification, 
                                negative values (red) push towards fraud classification.
                                """)

                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")
else:
    if app_mode != "Home" and data is None:
        st.warning("Please select a data source from the sidebar to continue.")