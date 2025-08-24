# SPDX-License-Identifier: MIT
# Copyright © 2025 github.com/dtiberio

# Heart Disease Prediction ML Pipeline - Training Script
# Following the comprehensive plan for Logistic Regression model training

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("HEART DISEASE PREDICTION - TRAINING PIPELINE")
print("=" * 60)

# =============================================================================
# 1. IMPORT LIBRARIES AND LOAD DATA
# =============================================================================
print("\n1. LOADING DATA...")
print("-" * 40)

# Load the dataset
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
print(f"Script directory: {script_dir}")
print(f"Parent directory: {parent_dir}")
df = pd.read_csv(os.path.join(parent_dir, 'heart.csv'))

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types and missing values:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n2. EXPLORATORY DATA ANALYSIS...")
print("-" * 40)

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Target variable distribution
print("\nTarget distribution:")
print(df['target'].value_counts())
print("\nTarget proportion:")
target_prop = df['target'].value_counts(normalize=True)
print(target_prop)

# Feature information
print("\nFeature descriptions:")
feature_descriptions = {
    'age': 'Age in years',
    'sex': 'Sex (1 = male; 0 = female)',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
    'restecg': 'Resting ECG results (0-2)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes; 0 = no)',
    'oldpeak': 'ST depression induced by exercise',
    'slope': 'Slope of peak exercise ST segment (0-2)',
    'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
    'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)',
    'target': 'Heart disease presence (1 = disease; 0 = no disease)'
}

for feature, description in feature_descriptions.items():
    print(f"{feature}: {description}")

# Create visualizations
plt.figure(figsize=(15, 12))

# Target distribution
plt.subplot(2, 3, 1)
df['target'].value_counts().plot(kind='bar', color=['lightcoral', 'lightblue'])
plt.title('Heart Disease Distribution')
plt.xlabel('Target (0=No Disease, 1=Disease)')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Age distribution by target
plt.subplot(2, 3, 2)
df[df['target'] == 0]['age'].hist(alpha=0.5, label='No Disease', bins=20, color='lightblue')
df[df['target'] == 1]['age'].hist(alpha=0.5, label='Disease', bins=20, color='lightcoral')
plt.title('Age Distribution by Target')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()

# Correlation heatmap
plt.subplot(2, 3, 3)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')

# Chest pain type distribution
plt.subplot(2, 3, 4)
pd.crosstab(df['cp'], df['target']).plot(kind='bar', color=['lightblue', 'lightcoral'])
plt.title('Chest Pain Type vs Target')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'])
plt.xticks(rotation=0)

# Max heart rate by target
plt.subplot(2, 3, 5)
df.boxplot(column='thalach', by='target', ax=plt.gca())
plt.title('Max Heart Rate by Target')
plt.xlabel('Target')
plt.ylabel('Max Heart Rate')

# Exercise angina vs target
plt.subplot(2, 3, 6)
pd.crosstab(df['exang'], df['target']).plot(kind='bar', color=['lightblue', 'lightcoral'])
plt.title('Exercise Angina vs Target')
plt.xlabel('Exercise Angina')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'heart_disease_eda.png'), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
print("\n3. DATA PREPROCESSING...")
print("-" * 40)

# Identify feature types
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# Check for outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers)

print("\nOutlier detection (IQR method):")
for col in numerical_features:
    outlier_count = detect_outliers_iqr(df, col)
    print(f"{col}: {outlier_count} outliers")

# =============================================================================
# 4. FEATURE SELECTION AND TARGET SEPARATION
# =============================================================================
print("\n4. FEATURE AND TARGET SEPARATION...")
print("-" * 40)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Feature names:", X.columns.tolist())

# =============================================================================
# 5. TRAIN-TEST SPLIT
# =============================================================================
print("\n5. TRAIN-TEST SPLIT...")
print("-" * 40)

# Split data with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))
print("\nTest target distribution:")
print(y_test.value_counts(normalize=True))

# =============================================================================
# 6. FEATURE SCALING
# =============================================================================
print("\n6. FEATURE SCALING...")
print("-" * 40)

# Initialize scaler
scaler = StandardScaler()

# Fit on training data only to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully")
print("Training features mean (should be ~0):", np.round(np.mean(X_train_scaled, axis=0), 4))
print("Training features std (should be ~1):", np.round(np.std(X_train_scaled, axis=0), 4))

# =============================================================================
# 7. INITIAL MODEL TRAINING
# =============================================================================
print("\n7. INITIAL MODEL TRAINING...")
print("-" * 40)

# Initialize Logistic Regression model
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000  # Increase if convergence issues
)

# Train the model
lr_model.fit(X_train_scaled, y_train)
print("Initial model trained successfully!")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# =============================================================================
# 8. CROSS-VALIDATION AND INITIAL EVALUATION
# =============================================================================
print("\n8. CROSS-VALIDATION AND INITIAL EVALUATION...")
print("-" * 40)

# Perform cross-validation
cv_scores = cross_val_score(
    lr_model, X_train_scaled, y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)

print("Cross-validation accuracy scores:", np.round(cv_scores, 4))
print("Mean CV accuracy:", np.round(cv_scores.mean(), 4))
print("CV accuracy std:", np.round(cv_scores.std(), 4))

# AUC-ROC cross-validation
cv_auc_scores = cross_val_score(
    lr_model, X_train_scaled, y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc'
)

print("\nCross-validation AUC scores:", np.round(cv_auc_scores, 4))
print("Mean CV AUC:", np.round(cv_auc_scores.mean(), 4))
print("CV AUC std:", np.round(cv_auc_scores.std(), 4))

# Training performance
y_train_pred = lr_model.predict(X_train_scaled)
y_train_prob = lr_model.predict_proba(X_train_scaled)[:, 1]

print("\nTraining Set Classification Report:")
print(classification_report(y_train, y_train_pred))

# Confusion matrix
train_cm = confusion_matrix(y_train, y_train_pred)
print("\nTraining Confusion Matrix:")
print(train_cm)

# AUC-ROC Score
train_auc = roc_auc_score(y_train, y_train_prob)
print(f"\nTraining AUC-ROC Score: {train_auc:.4f}")

# =============================================================================
# 9. HYPERPARAMETER TUNING
# =============================================================================
print("\n9. HYPERPARAMETER TUNING...")
print("-" * 40)

# Define parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # liblinear supports both l1 and l2
}

print("Parameter grid:", param_grid)

# Grid search with cross-validation
grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Performing grid search...")
grid_search.fit(X_train_scaled, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation AUC score:", np.round(grid_search.best_score_, 4))

# Use the best model
best_lr_model = grid_search.best_estimator_
print("Best model selected!")

# =============================================================================
# 10. FINAL PIPELINE CREATION
# =============================================================================
print("\n10. FINAL PIPELINE CREATION...")
print("-" * 40)

# Create a complete pipeline with best parameters
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        C=grid_search.best_params_['C'],
        penalty=grid_search.best_params_['penalty'],
        solver=grid_search.best_params_['solver'],
        random_state=42,
        max_iter=1000
    ))
])

# Fit final pipeline
final_pipeline.fit(X_train, y_train)
print("Final pipeline trained successfully!")

# Evaluate final pipeline with cross-validation
final_cv_scores = cross_val_score(
    final_pipeline, X_train, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc'
)

print("Final pipeline CV AUC scores:", np.round(final_cv_scores, 4))
print("Final pipeline mean CV AUC:", np.round(final_cv_scores.mean(), 4))
print("Final pipeline CV AUC std:", np.round(final_cv_scores.std(), 4))

# =============================================================================
# 11. MODEL INTERPRETATION
# =============================================================================
print("\n11. MODEL INTERPRETATION...")
print("-" * 40)

# Get the trained classifier from the pipeline
trained_classifier = final_pipeline.named_steps['classifier']

# Feature coefficients interpretation
coefficients = trained_classifier.coef_[0]
feature_names = X.columns

# Create interpretation dataframe
interpretation_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients),
    'Odds_Ratio': np.exp(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("Feature Interpretation (sorted by absolute coefficient):")
print(interpretation_df)

# Visualize feature importance
plt.figure(figsize=(12, 8))
plt.barh(range(len(interpretation_df)), interpretation_df['Coefficient'], 
         color=['red' if x < 0 else 'blue' for x in interpretation_df['Coefficient']])
plt.yticks(range(len(interpretation_df)), interpretation_df['Feature'])
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Feature Coefficients')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'feature_coefficients.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\nInterpretation Guide:")
print("- Positive coefficients increase the probability of heart disease")
print("- Negative coefficients decrease the probability of heart disease") 
print("- Larger absolute values indicate stronger influence")
print("- Odds ratio > 1: increases odds, < 1: decreases odds")

# =============================================================================
# 12. SAVE THE TRAINED MODEL
# =============================================================================
print("\n12. SAVING THE TRAINED MODEL...")
print("-" * 40)

# Save the trained pipeline
joblib.dump(final_pipeline, os.path.join(script_dir, 'heart_disease_model.pkl'))
print("✓ Model saved as 'heart_disease_model.pkl'")

# Save feature names for future use
joblib.dump(list(X.columns), os.path.join(script_dir, 'feature_names.pkl'))
print("✓ Feature names saved as 'feature_names.pkl'")

# Save model performance metrics
model_metrics = {
    'cv_auc_mean': final_cv_scores.mean(),
    'cv_auc_std': final_cv_scores.std(),
    'best_params': grid_search.best_params_,
    'feature_importance': interpretation_df.to_dict('records')
}

joblib.dump(model_metrics, os.path.join(script_dir, 'model_metrics.pkl'))
print("✓ Model metrics saved as 'model_metrics.pkl'")

# =============================================================================
# 13. TRAINING SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

print(f"✓ Dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"✓ Target distribution: {target_prop[1]:.1%} positive cases")
print(f"✓ Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
print(f"✓ Best hyperparameters: {grid_search.best_params_}")
print(f"✓ Cross-validation AUC: {final_cv_scores.mean():.4f} ± {final_cv_scores.std():.4f}")
print(f"✓ Most important features:")
for i, (_, row) in enumerate(interpretation_df.head(3).iterrows()):
    print(f"   {i+1}. {row['Feature']}: {row['Coefficient']:.4f}")

print("\n✓ Model files saved:")
print("   - heart_disease_model.pkl (complete pipeline)")
print("   - feature_names.pkl (feature list)")
print("   - model_metrics.pkl (performance metrics)")

print("\n✓ Visualization files created:")
print("   - heart_disease_eda.png (exploratory analysis)")
print("   - feature_coefficients.png (feature importance)")

print("\n🎯 The model is ready for inference and testing!")
print("   Next steps: Create inference script and evaluate on test set")

print("\n" + "=" * 60)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)