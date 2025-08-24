# SPDX-License-Identifier: MIT
# Copyright © 2025 github.com/dtiberio

#!/usr/bin/env python3
"""
Breast Cancer Prediction ML Training Pipeline
Based on Wisconsin Breast Cancer Dataset
Implements Steps 1-12 of the training plan
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_score, recall_score, f1_score)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("BREAST CANCER PREDICTION ML TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Import Libraries and Load Data
    print("\n" + "="*50)
    print("STEP 1: LOADING DATA")
    print("="*50)
    
    # Get the parent directory to find cancer.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, 'cancer.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"cancer.csv not found at {data_path}")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Step 2: Exploratory Data Analysis (EDA)
    print("\n" + "="*50)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Display basic information
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types and missing values:")
    print(df.info())
    
    print("\nStatistical summary:")
    print(df.describe().round(3))
    
    # Check for missing values
    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")
    
    # Target variable distribution
    print("\nDiagnosis distribution:")
    print(df['diagnosis'].value_counts())
    print("Diagnosis proportion:")
    print(df['diagnosis'].value_counts(normalize=True).round(3))
    
    # Check for duplicate records
    print(f"\nDuplicate records: {df.duplicated().sum()}")
    
    # Check ID column uniqueness
    print(f"Unique IDs: {df['id'].nunique()} out of {len(df)} samples")
    
    # Visualizations
    print("\nGenerating EDA visualizations...")
    
    # Create output directory for plots
    output_dir = os.path.join(current_dir, 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Target distribution and key feature analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Target distribution
    df['diagnosis'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'salmon'])
    axes[0,0].set_title('Diagnosis Distribution')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=0)
    
    # Key feature distributions by diagnosis
    key_features = ['radius_mean', 'texture_mean', 'area_mean']
    for i, feature in enumerate(key_features):
        ax = axes[0,1] if i == 0 else axes[1,0] if i == 1 else axes[1,1]
        df.boxplot(column=feature, by='diagnosis', ax=ax)
        ax.set_title(f'{feature} by Diagnosis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(20, 16))
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"EDA plots saved to: {output_dir}")
    
    # Step 3: Data Preprocessing
    print("\n" + "="*50)
    print("STEP 3: DATA PREPROCESSING")
    print("="*50)
    
    # 3.1 Remove Irrelevant Features
    print("3.1 Removing irrelevant features...")
    df_processed = df.drop('id', axis=1)
    print(f"Features after removing ID: {df_processed.shape[1]}")
    
    # Check for any unnamed columns (common in CSV exports)
    unnamed_cols = [col for col in df_processed.columns if 'Unnamed' in col]
    if unnamed_cols:
        df_processed = df_processed.drop(unnamed_cols, axis=1)
        print(f"Removed unnamed columns: {unnamed_cols}")
    
    # 3.2 Encode Target Variable
    print("\n3.2 Encoding target variable...")
    label_encoder = LabelEncoder()
    df_processed['diagnosis_encoded'] = label_encoder.fit_transform(df_processed['diagnosis'])
    
    print("Diagnosis encoding:")
    print("B (Benign) -> 0")
    print("M (Malignant) -> 1")
    print(f"Encoded distribution:\n{df_processed['diagnosis_encoded'].value_counts().sort_index()}")
    
    # 3.3 Feature Analysis and Selection
    print("\n3.3 Feature analysis...")
    # Identify feature groups
    mean_features = [col for col in df_processed.columns if col.endswith('_mean')]
    se_features = [col for col in df_processed.columns if col.endswith('_se')]
    worst_features = [col for col in df_processed.columns if col.endswith('_worst')]
    
    print(f"Mean features: {len(mean_features)}")
    print(f"SE features: {len(se_features)}")
    print(f"Worst features: {len(worst_features)}")
    
    # Check for highly correlated features (multicollinearity)
    def find_correlated_features(df, threshold=0.9):
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        
        high_corr_pairs = []
        for col in upper_tri.columns:
            high_corr_features = upper_tri.index[upper_tri[col] > threshold].tolist()
            for feature in high_corr_features:
                high_corr_pairs.append((feature, col, upper_tri.loc[feature, col]))
        
        return high_corr_pairs
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['diagnosis_encoded']]
    high_corr = find_correlated_features(df_processed[numeric_cols])
    
    print(f"\nHighly correlated feature pairs (>0.9): {len(high_corr)}")
    for feat1, feat2, corr in high_corr[:10]:  # Show first 10
        print(f"  {feat1} - {feat2}: {corr:.3f}")
    
    # 3.4 Handle Outliers
    print("\n3.4 Outlier analysis...")
    def detect_outliers_iqr(df, columns):
        outlier_info = {}
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outlier_info[column] = len(outliers)
        
        return outlier_info
    
    outlier_counts = detect_outliers_iqr(df_processed, numeric_cols)
    print("Outlier counts per feature (top 10):")
    sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
    for feature, count in sorted_outliers[:10]:
        if count > 0:
            print(f"  {feature}: {count} outliers")
    
    # Step 4: Feature Selection and Target Separation
    print("\n" + "="*50)
    print("STEP 4: FEATURE SELECTION AND TARGET SEPARATION")
    print("="*50)
    
    # Separate features and target
    feature_columns = [col for col in df_processed.columns 
                      if col not in ['diagnosis', 'diagnosis_encoded']]
    X = df_processed[feature_columns]
    y = df_processed['diagnosis_encoded']
    
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Feature names:", len(X.columns), "features")
    
    # Feature importance using correlation with target
    feature_target_corr = X.corrwith(y).abs().sort_values(ascending=False)
    print("\nTop 10 features correlated with target:")
    print(feature_target_corr.head(10).round(3))
    
    # Step 5: Train-Test Split
    print("\n" + "="*50)
    print("STEP 5: TRAIN-TEST SPLIT")
    print("="*50)
    
    # Split data with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print("Training set size:", X_train.shape)
    print("Test set size:", X_test.shape)
    print("Training target distribution:")
    print(y_train.value_counts(normalize=True).round(3))
    print("Test target distribution:")
    print(y_test.value_counts(normalize=True).round(3))
    
    # Step 6: Feature Scaling
    print("\n" + "="*50)
    print("STEP 6: FEATURE SCALING")
    print("="*50)
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on training data only to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled successfully")
    print("Training features mean (should be ~0):", np.mean(X_train_scaled, axis=0)[:5].round(3))
    print("Training features std (should be ~1):", np.std(X_train_scaled, axis=0)[:5].round(3))
    
    # Convert back to DataFrame for easier handling
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    # Step 7: Model Training
    print("\n" + "="*50)
    print("STEP 7: MODEL TRAINING")
    print("="*50)
    
    # 7.1 Initial Model Training
    print("7.1 Initial model training...")
    # Initialize Logistic Regression model
    # Using class_weight='balanced' to handle class imbalance
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Train the model
    lr_model.fit(X_train_scaled, y_train)
    print("Model trained successfully!")
    
    # Check if model converged
    if lr_model.n_iter_[0] == lr_model.max_iter:
        print("Warning: Model may not have converged. Consider increasing max_iter.")
    else:
        print(f"Model converged after {lr_model.n_iter_[0]} iterations")
    
    # 7.2 Feature Importance Analysis
    print("\n7.2 Feature importance analysis...")
    # Get feature coefficients (importance)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lr_model.coef_[0],
        'abs_coefficient': np.abs(lr_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("Top 15 most important features:")
    print(feature_importance.head(15).round(3))
    
    # Visualize top features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    colors = ['red' if x > 0 else 'blue' for x in top_features['coefficient'][::-1]]
    plt.barh(range(len(top_features)), top_features['coefficient'][::-1], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'][::-1])
    plt.xlabel('Coefficient Value')
    plt.title('Top 15 Feature Coefficients (Logistic Regression)')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Step 8: Model Evaluation on Training Data
    print("\n" + "="*50)
    print("STEP 8: MODEL EVALUATION ON TRAINING DATA")
    print("="*50)
    
    # 8.1 Cross-Validation
    print("8.1 Cross-validation...")
    # Perform cross-validation
    cv_scores = cross_val_score(
        lr_model, X_train_scaled, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc'  # Use AUC for imbalanced dataset
    )
    
    cv_accuracy = cross_val_score(
        lr_model, X_train_scaled, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    
    print("Cross-validation AUC scores:", cv_scores.round(3))
    print("Mean CV AUC:", cv_scores.mean().round(3))
    print("CV AUC std:", cv_scores.std().round(3))
    print("Mean CV Accuracy:", cv_accuracy.mean().round(3))
    
    # 8.2 Training Performance
    print("\n8.2 Training performance...")
    # Predictions on training set
    y_train_pred = lr_model.predict(X_train_scaled)
    y_train_prob = lr_model.predict_proba(X_train_scaled)[:, 1]
    
    # Classification report
    print("Training Set Classification Report:")
    print(classification_report(y_train, y_train_pred, 
                              target_names=['Benign', 'Malignant']))
    
    # Confusion matrix
    train_cm = confusion_matrix(y_train, y_train_pred)
    print("Training Confusion Matrix:")
    print(train_cm)
    
    # Additional metrics for imbalanced dataset
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    # AUC-ROC Score
    train_auc = roc_auc_score(y_train, y_train_prob)
    print(f"\nTraining Metrics:")
    print(f"AUC-ROC Score: {train_auc:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall (Sensitivity): {train_recall:.4f}")
    print(f"F1-Score: {train_f1:.4f}")
    
    # Step 9: Hyperparameter Tuning
    print("\n" + "="*50)
    print("STEP 9: HYPERPARAMETER TUNING")
    print("="*50)
    
    # Define parameter grid for Logistic Regression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced']
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=2000),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',  # Primary metric for imbalanced data
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train_scaled, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation AUC:", f"{grid_search.best_score_:.4f}")
    
    # Use the best model
    best_lr_model = grid_search.best_estimator_
    
    # Step 10: Pipeline Creation
    print("\n" + "="*50)
    print("STEP 10: PIPELINE CREATION")
    print("="*50)
    
    # Create a complete pipeline with the best parameters
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            **grid_search.best_params_,
            random_state=42,
            max_iter=2000
        ))
    ])
    
    # Fit pipeline on full training data
    final_pipeline.fit(X_train, y_train)
    print("Final pipeline trained successfully!")
    
    # Evaluate pipeline on training data
    train_pipeline_pred = final_pipeline.predict(X_train)
    train_pipeline_prob = final_pipeline.predict_proba(X_train)[:, 1]
    train_pipeline_auc = roc_auc_score(y_train, train_pipeline_prob)
    
    print(f"Pipeline training AUC: {train_pipeline_auc:.4f}")
    
    # Step 11: Model Interpretation
    print("\n" + "="*50)
    print("STEP 11: MODEL INTERPRETATION")
    print("="*50)
    
    # Feature coefficients interpretation
    final_classifier = final_pipeline.named_steps['classifier']
    coefficients = final_classifier.coef_[0]
    feature_names = X.columns
    
    # Create interpretation dataframe
    interpretation_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients),
        'Odds_Ratio': np.exp(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("Feature Interpretation (Top 10):")
    print(interpretation_df.head(10).round(3))
    
    # Clinical interpretation
    print("\nClinical Interpretation:")
    print("Positive coefficients increase malignancy probability")
    print("Negative coefficients decrease malignancy probability")
    print("\nTop risk factors (positive coefficients):")
    risk_factors = interpretation_df[interpretation_df['Coefficient'] > 0].head(5)
    for _, row in risk_factors.iterrows():
        print(f"  {row['Feature']}: OR = {row['Odds_Ratio']:.3f}")
    
    print("\nTop protective factors (negative coefficients):")
    protective_factors = interpretation_df[interpretation_df['Coefficient'] < 0].head(5)
    for _, row in protective_factors.iterrows():
        print(f"  {row['Feature']}: OR = {row['Odds_Ratio']:.3f}")
    
    # Step 12: Save the Trained Model
    print("\n" + "="*50)
    print("STEP 12: SAVE TRAINED MODEL")
    print("="*50)
    
    # Create directory for model artifacts (already created)
    models_dir = output_dir
    
    # Save the trained pipeline
    model_path = os.path.join(models_dir, 'breast_cancer_model.pkl')
    feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
    metrics_path = os.path.join(models_dir, 'model_metrics.pkl')
    
    joblib.dump(final_pipeline, model_path)
    joblib.dump(list(X.columns), feature_names_path)
    
    # Save training metrics and model info
    model_metrics = {
        'best_params': grid_search.best_params_,
        'cv_auc_mean': grid_search.best_score_, 
        'cv_auc_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
        'train_auc': train_pipeline_auc,
        'feature_importance': interpretation_df.to_dict('records'),
        'class_distribution': y.value_counts().to_dict(),
        'label_encoding': {'B': 0, 'M': 1}
    }
    
    joblib.dump(model_metrics, metrics_path)
    
    print("Model artifacts saved successfully!")
    print(f"  - Model: {model_path}")
    print(f"  - Features: {feature_names_path}")
    print(f"  - Metrics: {metrics_path}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final Model Performance:")
    print(f"  - Best CV AUC: {grid_search.best_score_:.4f}")
    print(f"  - Training AUC: {train_pipeline_auc:.4f}")
    print(f"  - Best Parameters: {grid_search.best_params_}")
    print(f"  - Total Features: {len(X.columns)}")
    print(f"  - Training Samples: {len(X_train)}")
    print(f"  - Test Samples: {len(X_test)}")
    print("\nFiles saved:")
    print(f"  - Model artifacts: {models_dir}")
    print(f"  - Visualizations: {models_dir}")
    
    return final_pipeline, model_metrics, X_test, y_test

if __name__ == "__main__":
    pipeline, metrics, X_test, y_test = main()
    print("\nTraining pipeline completed successfully!")