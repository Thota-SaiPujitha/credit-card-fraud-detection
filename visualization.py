import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def plot_class_distribution(data):
    """
    Plot the distribution of classes in the dataset.

    Parameters:
    -----------
    data : DataFrame
        The input data with 'Class' column

    Returns:
    --------
    fig : Figure
        The matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count classes
    class_counts = data['Class'].value_counts().sort_index()
    labels = ['Normal', 'Fraud']

    # Create the bar plot
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, legend=False, ax=ax,
                palette='viridis')

    # Add values on top of the bars
    for i, count in enumerate(class_counts.values):
        ax.text(i, count + (max(class_counts.values) * 0.01), f"{count:,}",
                ha='center', fontsize=10)

    # Calculate percentages
    total = len(data)
    normal_pct = class_counts.get(0, 0) / total * 100
    fraud_pct = class_counts.get(1, 0) / total * 100

    # Add percentage labels
    ax.text(0, class_counts.get(0, 0) - (max(class_counts.values) * 0.1),
            f"{normal_pct:.2f}%", ha='center', fontsize=12, color='white')

    ax.text(1, class_counts.get(1, 0) + (max(class_counts.values) * 0.01),
            f"{fraud_pct:.2f}%", ha='center', fontsize=12)

    # Add a second y-axis for percentages
    secax = ax.secondary_yaxis('right', functions=(lambda x: x / total * 100, lambda x: x * total / 100))
    secax.set_ylabel('Percentage (%)')

    # Customize the plot
    ax.set_title('Distribution of Normal vs Fraudulent Transactions', fontsize=14)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)

    # Set tick positions first, then labels (to avoid warning)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)

    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_correlation_matrix(data):
    """
    Plot a correlation matrix for the features.

    Parameters:
    -----------
    data : DataFrame
        The input data with features

    Returns:
    --------
    fig : Figure
        The matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Create the heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=True, fmt=".2f",
                linewidths=0.5, vmin=-1, vmax=1, ax=ax)

    # Customize the plot
    ax.set_title('Feature Correlation Matrix', fontsize=14)

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_feature_importance(feature_importance, top_n=15):
    """
    Plot feature importance.

    Parameters:
    -----------
    feature_importance : dict
        Dictionary mapping feature names to importance values
    top_n : int
        Number of top features to show

    Returns:
    --------
    fig : Figure
        The matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert to DataFrame for easier handling
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })

    # Sort by importance and take top N
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)

    # Create the bar plot
    sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', legend=False, palette='viridis', ax=ax)

    # Customize the plot
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot a confusion matrix.

    Parameters:
    -----------
    model : estimator
        The trained model
    X_test : DataFrame
        Testing features
    y_test : Series
        Testing labels

    Returns:
    --------
    fig : Figure
        The matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create the heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    # Customize the plot
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)

    # Set tick positions first, then labels (to avoid warning)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Fraud'])
    ax.set_yticklabels(['Normal', 'Fraud'])

    # Add TN, FP, FN, TP labels
    tn, fp, fn, tp = cm.ravel()

    # Add percentages
    total = y_test.shape[0]
    ax.text(0, 0, f"{tn}\n({tn / total:.1%})", ha='center', va='center', fontsize=12)
    ax.text(1, 0, f"{fp}\n({fp / total:.1%})", ha='center', va='center', fontsize=12)
    ax.text(0, 1, f"{fn}\n({fn / total:.1%})", ha='center', va='center', fontsize=12)
    ax.text(1, 1, f"{tp}\n({tp / total:.1%})", ha='center', va='center', fontsize=12)

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_roc_curve(model, X_test, y_test):
    """
    Plot a ROC curve.

    Parameters:
    -----------
    model : estimator
        The trained model
    X_test : DataFrame
        Testing features
    y_test : Series
        Testing labels

    Returns:
    --------
    fig : Figure
        The matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    # Customize the plot
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(loc='lower right')

    # Add grid
    ax.grid(linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_precision_recall_curve(model, X_test, y_test):
    """
    Plot a precision-recall curve.

    Parameters:
    -----------
    model : estimator
        The trained model
    X_test : DataFrame
        Testing features
    y_test : Series
        Testing labels

    Returns:
    --------
    fig : Figure
        The matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # Calculate average precision
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_test, y_pred_proba)

    # Plot precision-recall curve
    ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap:.3f})')

    # Calculate no-skill line (the prevalence of the positive class)
    no_skill = sum(y_test) / len(y_test)
    ax.plot([0, 1], [no_skill, no_skill], color='navy', lw=2, linestyle='--', label='No Skill')

    # Customize the plot
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower left')

    # Add grid
    ax.grid(linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    return fig
