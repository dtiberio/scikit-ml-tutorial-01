<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright © 2025 github.com/dtiberio
-->

# Breast Cancer Prediction ML Pipeline - Training Plan

This tutorial outlines a comprehensive machine learning pipeline for predicting breast cancer diagnosis using Logistic Regression on the Wisconsin Breast Cancer dataset.

## Dataset Overview

The cancer.csv dataset contains 569 records with 32 features:

- **ID:** Unique sample identifier (to be excluded from modeling)
- **Diagnosis:** Target variable (M=Malignant, B=Benign)
- **30 Numeric Features:** Computed from cell nuclei images, grouped into:
  - **Mean values (10 features):** radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean
  - **Standard error values (10 features):** radius_se, texture_se, etc.
  - **Worst/largest values (10 features):** radius_worst, texture_worst, etc.

**Class Distribution:** 357 Benign (62.7%), 212 Malignant (37.3%) - moderately imbalanced

## ML Pipeline - Training Plan

### Training Phase (Steps 1-12):

- Handles 30 numeric features (mean/SE/worst values)
- Addresses class imbalance (62.7% benign, 37.3% malignant)
- Uses class_weight='balanced' for LogisticRegression
- Includes multicollinearity analysis for correlated features
- Focuses on AUC-ROC as primary metric for imbalanced data

### Testing Phase (Steps 13-22):

- Comprehensive clinical metrics (sensitivity, specificity, PPV, NPV)
- Threshold optimization for clinical objectives
- Error analysis with focus on false negatives (missed cancers)
- Clinical interpretation emphasizing cancer detection priority

### TUI Application (Steps 23-29):

- Breast cancer-specific interface with tumor characteristics
- 5-tier malignancy risk classification (Very Low → Very High)
- Feature contribution analysis for cytological measurements
- Clinical recommendations appropriate for oncology workflows
- Feature imputation for missing measurements

### Cancer-Specific Features:

- Medical disclaimer for histopathological diagnosis
- Tumor characteristics collection (radius, texture, perimeter, area, etc.)
- Clinical decision support for pathologists/oncologists
- Focus on minimizing false negatives (missing malignant cases)

## Step-by-Step ML Pipeline

### 1. Import Libraries and Load Data

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('cancer.csv')
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
```

### 2. Exploratory Data Analysis (EDA)

**Purpose:** Understand data structure, feature relationships, and class distribution.

```python
# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types and missing values:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Target variable distribution
print("\nDiagnosis distribution:")
print(df['diagnosis'].value_counts())
print("Diagnosis proportion:")
print(df['diagnosis'].value_counts(normalize=True))

# Check for duplicate records
print(f"\nDuplicate records: {df.duplicated().sum()}")

# Check ID column uniqueness
print(f"Unique IDs: {df['id'].nunique()} out of {len(df)} samples")
```

**Visualization:**

- Histograms for feature distributions
- Box plots comparing malignant vs benign for key features
- Correlation heatmap between features
- Pair plots for highly correlated features
- Class distribution analysis

```python
# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Target distribution
df['diagnosis'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'salmon'])
axes[0,0].set_title('Diagnosis Distribution')
axes[0,0].set_ylabel('Count')

# Key feature distributions by diagnosis
key_features = ['radius_mean', 'texture_mean', 'area_mean', 'concavity_mean']
for i, feature in enumerate(key_features[:3]):
    ax = axes[0,1] if i == 0 else axes[1,0] if i == 1 else axes[1,1]
    df.boxplot(column=feature, by='diagnosis', ax=ax)
    ax.set_title(f'{feature} by Diagnosis')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(20, 16))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0,
            square=True, linewidths=0.1)
plt.title('Feature Correlation Heatmap')
plt.show()
```

### 3. Data Preprocessing

#### 3.1 Remove Irrelevant Features

```python
# Remove ID column as it's just an identifier
df_processed = df.drop('id', axis=1)
print(f"Features after removing ID: {df_processed.shape[1]}")

# Check for any unnamed columns (common in CSV exports)
unnamed_cols = [col for col in df_processed.columns if 'Unnamed' in col]
if unnamed_cols:
    df_processed = df_processed.drop(unnamed_cols, axis=1)
    print(f"Removed unnamed columns: {unnamed_cols}")
```

#### 3.2 Encode Target Variable

```python
# Convert diagnosis to numeric (M=1, B=0)
label_encoder = LabelEncoder()
df_processed['diagnosis_encoded'] = label_encoder.fit_transform(df_processed['diagnosis'])

print("Diagnosis encoding:")
print("B (Benign) -> 0")
print("M (Malignant) -> 1")
print(f"Encoded distribution: {df_processed['diagnosis_encoded'].value_counts().sort_index()}")
```

#### 3.3 Feature Analysis and Selection

```python
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
```

#### 3.4 Handle Outliers

```python
# Detect outliers using IQR method
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
print("\nOutlier counts per feature:")
for feature, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  {feature}: {count} outliers")
```

### 4. Feature Selection and Target Separation

```python
# Separate features and target
feature_columns = [col for col in df_processed.columns
                  if col not in ['diagnosis', 'diagnosis_encoded']]
X = df_processed[feature_columns]
y = df_processed['diagnosis_encoded']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Feature names:", X.columns.tolist())

# Feature importance using correlation with target
feature_target_corr = X.corrwith(y).abs().sort_values(ascending=False)
print("\nTop 10 features correlated with target:")
print(feature_target_corr.head(10))
```

### 5. Train-Test Split

```python
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
print(y_train.value_counts(normalize=True))
print("Test target distribution:")
print(y_test.value_counts(normalize=True))
```

### 6. Feature Scaling

**Why:** Logistic Regression is sensitive to feature scales, especially important with features having different units/ranges.

```python
# Initialize scaler
scaler = StandardScaler()

# Fit on training data only to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully")
print("Training features mean (should be ~0):", np.mean(X_train_scaled, axis=0)[:5])
print("Training features std (should be ~1):", np.std(X_train_scaled, axis=0)[:5])

# Convert back to DataFrame for easier handling
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
```

### 7. Model Training

#### 7.1 Initial Model Training

```python
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
```

#### 7.2 Feature Importance Analysis

```python
# Get feature coefficients (importance)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("Top 15 most important features:")
print(feature_importance.head(15))

# Visualize top features
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['coefficient'][::-1])
plt.yticks(range(len(top_features)), top_features['feature'][::-1])
plt.xlabel('Coefficient Value')
plt.title('Top 15 Feature Coefficients (Logistic Regression)')
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

### 8. Model Evaluation on Training Data

#### 8.1 Cross-Validation

```python
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

print("Cross-validation AUC scores:", cv_scores)
print("Mean CV AUC:", cv_scores.mean())
print("CV AUC std:", cv_scores.std())
print("Mean CV Accuracy:", cv_accuracy.mean())
```

#### 8.2 Training Performance

```python
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
from sklearn.metrics import precision_score, recall_score, f1_score
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
```

### 9. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

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
print("Best cross-validation AUC:", grid_search.best_score_)

# Use the best model
best_lr_model = grid_search.best_estimator_
```

### 10. Pipeline Creation (Best Practice)

```python
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
```

### 11. Model Interpretation

```python
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
print(interpretation_df.head(10))

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
```

### 12. Save the Trained Model

```python
import joblib
import os

# Create directory for model artifacts
os.makedirs('cancer_claude', exist_ok=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, 'cancer_claude')

# Save the trained pipeline
model_path = os.path.join(save_dir, 'breast_cancer_model.pkl')
feature_names_path = os.path.join(save_dir, 'feature_names.pkl')
metrics_path = os.path.join(save_dir, 'model_metrics.pkl')

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
```

## Key Considerations for Breast Cancer Classification

1. **Class Imbalance:** Use balanced class weights and AUC-ROC as primary metric
2. **Feature Scaling:** Critical due to different measurement scales
3. **Multicollinearity:** High correlation between mean, SE, and worst features
4. **Clinical Relevance:** Minimize false negatives (missing malignant cases)
5. **Feature Groups:** Consider using feature selection to reduce redundancy
6. **Interpretability:** Important for clinical acceptance and understanding

## Expected Outputs

After completing this pipeline, you should have:

- A trained and tuned Logistic Regression model for cancer prediction
- Feature importance rankings with clinical interpretation
- Cross-validation performance metrics
- A complete preprocessing pipeline
- Saved model files for inference

## Next Steps: Model Testing and Evaluation Plan

This training plan prepares the foundation for comprehensive model testing on breast cancer data.

---

## Model Testing Plan

### 13. Load Trained Model and Test Data Preparation

**Purpose:** Load the saved model and prepare test data for evaluation.

```python
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained pipeline and metadata
model_path = 'breast_cancer_model.pkl'
feature_names_path = 'feature_names.pkl'
metrics_path = 'model_metrics.pkl'

pipeline = joblib.load(model_path)
feature_names = joblib.load(feature_names_path)
model_metrics = joblib.load(metrics_path)

print("Model loaded successfully!")
print("Best training parameters:", model_metrics['best_params'])
print("Training CV AUC:", f"{model_metrics['cv_auc_mean']:.4f} ± {model_metrics['cv_auc_std']:.4f}")
print("Label encoding:", model_metrics['label_encoding'])
```

### 14. Test Set Predictions

**Purpose:** Generate predictions and probabilities on the held-out test set.

```python
# Make predictions on test set
y_test_pred = pipeline.predict(X_test)
y_test_prob = pipeline.predict_proba(X_test)[:, 1]

print("Test predictions generated")
print("Prediction distribution:")
print(f"  Benign (0): {np.sum(y_test_pred == 0)}")
print(f"  Malignant (1): {np.sum(y_test_pred == 1)}")
print(f"Probability range: {y_test_prob.min():.3f} - {y_test_prob.max():.3f}")

# Compare with actual distribution
print(f"\nActual test distribution:")
print(f"  Benign (0): {np.sum(y_test == 0)}")
print(f"  Malignant (1): {np.sum(y_test == 1)}")
```

### 15. Comprehensive Test Set Evaluation

#### 15.1 Classification Metrics

```python
# Calculate all classification metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)  # Sensitivity
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)

print("=== TEST SET PERFORMANCE ===")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall (Sensitivity): {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"AUC-ROC:   {test_auc:.4f}")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred,
                          target_names=['Benign', 'Malignant']))
```

#### 15.2 Confusion Matrix Analysis

```python
# Generate and visualize confusion matrix
test_cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(test_cm)

# Calculate additional metrics from confusion matrix
tn, fp, fn, tp = test_cm.ravel()
specificity = tn / (tn + fp)  # True negative rate
sensitivity = tp / (tp + fn)  # True positive rate (recall)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

print(f"\nClinical Metrics:")
print(f"Sensitivity (True Positive Rate):  {sensitivity:.4f}")
print(f"Specificity (True Negative Rate):  {specificity:.4f}")
print(f"Positive Predictive Value (PPV):   {ppv:.4f}")
print(f"Negative Predictive Value (NPV):   {npv:.4f}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title('Test Set Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nError Analysis:")
print(f"False Negatives (missed malignant): {fn} cases")
print(f"False Positives (false alarms): {fp} cases")
```

### 16. ROC Curve and Precision-Recall Analysis

#### 16.1 ROC Curve

```python
# Generate ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_prob)

plt.figure(figsize=(12, 5))

# ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {test_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve - Test Set')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
```

#### 16.2 Precision-Recall Curve

```python
# Generate Precision-Recall curve
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_test_prob)
from sklearn.metrics import auc
pr_auc = auc(recall_curve, precision_curve)

plt.subplot(1, 2, 2)
plt.plot(recall_curve, precision_curve, color='blue', lw=2,
         label=f'PR curve (AUC = {pr_auc:.3f})')

# Baseline precision (proportion of positive class)
baseline_precision = np.sum(y_test) / len(y_test)
plt.axhline(y=baseline_precision, color='red', linestyle='--',
            label=f'Baseline ({baseline_precision:.3f})')

plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (PPV)')
plt.title('Precision-Recall Curve - Test Set')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('test_roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 17. Threshold Analysis and Optimization

**Purpose:** Find optimal prediction threshold for different clinical objectives.

```python
# Calculate metrics for different thresholds
thresholds_analysis = []
for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_thresh = (y_test_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)

    # Calculate specificity
    cm = confusion_matrix(y_test, y_pred_thresh)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spec = 0

    thresholds_analysis.append({
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1_score': f1,
        'youden_index': rec + spec - 1  # J-statistic
    })

threshold_df = pd.DataFrame(thresholds_analysis)

# Find optimal thresholds for different objectives
opt_f1_thresh = threshold_df.loc[threshold_df['f1_score'].idxmax(), 'threshold']
opt_youden_thresh = threshold_df.loc[threshold_df['youden_index'].idxmax(), 'threshold']
opt_sensitivity_thresh = threshold_df.loc[threshold_df['recall'].idxmax(), 'threshold']

print("Optimal Thresholds:")
print(f"Best F1-Score threshold: {opt_f1_thresh:.2f}")
print(f"Best Youden Index threshold: {opt_youden_thresh:.2f}")
print(f"Best Sensitivity threshold: {opt_sensitivity_thresh:.2f}")

# Clinical consideration: prioritize sensitivity (minimize false negatives)
high_sensitivity_mask = threshold_df['recall'] >= 0.95
if len(threshold_df[high_sensitivity_mask]) > 0:
    clinical_thresh = threshold_df[high_sensitivity_mask]['threshold'].min()
    print(f"Clinical threshold (≥95% sensitivity): {clinical_thresh:.2f}")

print(f"\nDefault threshold (0.5) metrics:")
default_metrics = threshold_df[abs(threshold_df['threshold'] - 0.5) < 0.01]
if len(default_metrics) > 0:
    print(default_metrics[['threshold', 'accuracy', 'precision', 'recall', 'f1_score']].iloc[0])
```

### 18. Feature Importance Validation on Test Set

**Purpose:** Verify that important features remain consistent and interpretable.

```python
# Get feature coefficients from trained pipeline
trained_classifier = pipeline.named_steps['classifier']
test_coefficients = trained_classifier.coef_[0]

# Create feature importance dataframe
test_feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': test_coefficients,
    'Abs_Coefficient': np.abs(test_coefficients),
    'Odds_Ratio': np.exp(test_coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("Feature Importance (Test Model - Top 15):")
print(test_feature_importance.head(15))

# Compare with training feature importance
training_importance = pd.DataFrame(model_metrics['feature_importance'])
print("\nTop 10 Most Important Features:")
for i, (_, row) in enumerate(test_feature_importance.head(10).iterrows()):
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    significance = "strongly" if abs(row['Coefficient']) > 1 else "moderately"
    print(f"{i+1:2d}. {row['Feature']:20s}: {significance} {direction} malignancy risk "
          f"(coef: {row['Coefficient']:+.3f}, OR: {row['Odds_Ratio']:.3f})")
```

### 19. Model Performance Comparison

**Purpose:** Compare training vs. test performance to check for overfitting.

```python
# Compare training and test metrics
comparison_metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Training_CV': [
        'N/A',  # CV accuracy not stored in original plan
        'N/A',  # CV precision not stored
        'N/A',  # CV recall not stored
        'N/A',  # CV F1 not stored
        model_metrics['cv_auc_mean']
    ],
    'Training_Final': [
        'N/A',  # Would need to recalculate
        'N/A',  # Would need to recalculate
        'N/A',  # Would need to recalculate
        'N/A',  # Would need to recalculate
        model_metrics.get('train_auc', 'N/A')
    ],
    'Test_Set': [test_accuracy, test_precision, test_recall, test_f1, test_auc]
}

comparison_df = pd.DataFrame(comparison_metrics)
print("=== TRAINING vs TEST PERFORMANCE ===")
print(comparison_df)

# Check for overfitting using AUC
if 'train_auc' in model_metrics:
    auc_diff = model_metrics['train_auc'] - test_auc
    cv_auc_diff = model_metrics['cv_auc_mean'] - test_auc
else:
    cv_auc_diff = model_metrics['cv_auc_mean'] - test_auc

print(f"\nGeneralization Analysis:")
print(f"CV AUC vs Test AUC difference: {cv_auc_diff:.3f}")

if abs(cv_auc_diff) < 0.05:
    print("✓ Excellent generalization (difference < 0.05)")
elif abs(cv_auc_diff) < 0.1:
    print("✓ Good generalization (difference < 0.10)")
elif cv_auc_diff > 0.1:
    print("⚠ Possible overfitting (CV >> Test)")
else:
    print("⚠ Possible underfitting (Test >> CV)")
```

### 20. Error Analysis

**Purpose:** Analyze misclassified cases to understand model limitations.

```python
# Identify misclassified samples
misclassified_mask = y_test != y_test_pred
misclassified_indices = y_test.index[misclassified_mask]

print(f"Misclassified samples: {len(misclassified_indices)} out of {len(y_test)} ({len(misclassified_indices)/len(y_test):.1%})")

# Analyze false positives and false negatives
false_positives = y_test.index[(y_test == 0) & (y_test_pred == 1)]
false_negatives = y_test.index[(y_test == 1) & (y_test_pred == 0)]

print(f"\nError Breakdown:")
print(f"False Positives (Benign predicted as Malignant): {len(false_positives)}")
print(f"False Negatives (Malignant predicted as Benign): {len(false_negatives)}")

# Examine characteristics of misclassified cases
if len(false_positives) > 0:
    print(f"\n=== FALSE POSITIVE ANALYSIS ===")
    fp_data = X_test.loc[false_positives]
    fp_probs = y_test_prob[y_test.index.get_indexer(false_positives)]

    print("False Positive Characteristics (mean values):")
    fp_summary = fp_data.describe().loc['mean'].sort_values(ascending=False)
    print(fp_summary.head(10))

    print(f"\nFalse Positive Probabilities:")
    print(f"Mean: {fp_probs.mean():.3f}, Range: {fp_probs.min():.3f} - {fp_probs.max():.3f}")

if len(false_negatives) > 0:
    print(f"\n=== FALSE NEGATIVE ANALYSIS ===")
    fn_data = X_test.loc[false_negatives]
    fn_probs = y_test_prob[y_test.index.get_indexer(false_negatives)]

    print("False Negative Characteristics (mean values):")
    fn_summary = fn_data.describe().loc['mean'].sort_values(ascending=False)
    print(fn_summary.head(10))

    print(f"\nFalse Negative Probabilities:")
    print(f"Mean: {fn_probs.mean():.3f}, Range: {fn_probs.min():.3f} - {fn_probs.max():.3f}")
```

### 21. Clinical Interpretation

**Purpose:** Interpret results from a medical/clinical perspective.

```python
print("=== CLINICAL INTERPRETATION ===")
print(f"\nModel Performance Summary:")
print(f"• Sensitivity (Cancer Detection Rate): {test_recall:.1%}")
print(f"  - The model correctly identifies {test_recall:.1%} of malignant cases")
print(f"  - {fn} malignant cases were missed (False Negatives)")

print(f"\n• Specificity (Benign Identification): {specificity:.1%}")
print(f"  - The model correctly identifies {specificity:.1%} of benign cases")
print(f"  - {fp} benign cases were incorrectly flagged (False Positives)")

print(f"\n• Positive Predictive Value: {ppv:.1%}")
print(f"  - Of cases predicted as malignant, {ppv:.1%} actually are malignant")

print(f"\n• Negative Predictive Value: {npv:.1%}")
print(f"  - Of cases predicted as benign, {npv:.1%} actually are benign")

print(f"\n• Overall Accuracy: {test_accuracy:.1%}")
print(f"• AUC-ROC: {test_auc:.3f} ({'Excellent' if test_auc > 0.9 else 'Good' if test_auc > 0.8 else 'Fair' if test_auc > 0.7 else 'Poor'} discrimination)")

print(f"\nClinical Impact Assessment:")
if fn > 0:
    print(f"⚠ Critical: {fn} malignant cases missed by model")
    print(f"  These patients would not receive immediate cancer treatment")
if fp > 0:
    print(f"⚠ {fp} benign cases incorrectly flagged as malignant")
    print(f"  These patients would undergo unnecessary further testing/anxiety")

print(f"\nTop Malignancy Risk Indicators:")
malignancy_factors = test_feature_importance[test_feature_importance['Coefficient'] > 0].head(5)
for i, (_, row) in enumerate(malignancy_factors.iterrows()):
    print(f"  {i+1}. {row['Feature']}: {row['Odds_Ratio']:.2f}x increased odds")

print(f"\nTop Benign Indicators:")
benign_factors = test_feature_importance[test_feature_importance['Coefficient'] < 0].head(3)
for i, (_, row) in enumerate(benign_factors.iterrows()):
    print(f"  {i+1}. {row['Feature']}: {row['Odds_Ratio']:.2f}x odds (protective)")
```

### 22. Save Test Results

**Purpose:** Persist test results for reporting and future reference.

```python
# Save comprehensive test results
test_results = {
    'test_metrics': {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': test_f1,
        'auc_roc': test_auc,
        'positive_predictive_value': ppv,
        'negative_predictive_value': npv
    },
    'confusion_matrix': test_cm.tolist(),
    'optimal_thresholds': {
        'best_f1': opt_f1_thresh,
        'best_youden': opt_youden_thresh,
        'default': 0.5
    },
    'feature_importance': test_feature_importance.to_dict('records'),
    'misclassification_analysis': {
        'total_misclassified': len(misclassified_indices),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
    },
    'threshold_analysis': threshold_df.to_dict('records')
}

joblib.dump(test_results, 'test_results.pkl')
print("✓ Test results saved as 'test_results.pkl'")

print("\n=== TESTING COMPLETED ===")
print("All test artifacts saved:")
print("  - test_confusion_matrix.png")
print("  - test_roc_pr_curves.png")
print("  - test_results.pkl")
```

## Testing Summary Checklist

After completing the testing phase, verify:

- [ ] Model loads correctly from saved files
- [ ] Test set performance indicates good generalization
- [ ] Confusion matrix shows acceptable false negative rate
- [ ] ROC-AUC demonstrates good discrimination ability
- [ ] Optimal thresholds identified for clinical use
- [ ] Feature importance validated and clinically meaningful
- [ ] Error analysis completed for misclassified cases
- [ ] Clinical interpretation provided with actionable insights
- [ ] All results and visualizations saved

## Key Testing Considerations for Cancer Prediction

1. **Clinical Priority:** Minimize false negatives (missing cancer cases)
2. **Threshold Selection:** Consider setting lower threshold to increase sensitivity
3. **Feature Interpretability:** Ensure most important features make clinical sense
4. **Performance Stability:** Test performance should be close to CV performance
5. **Error Analysis:** Understand what characteristics lead to misclassification

---

## Model Inference Plan - Interactive TUI Application

After training and testing the breast cancer prediction model, create a professional clinical interface for real-world inference using Python's `rich` library.

### 23. Interactive Inference Application Setup

**Purpose:** Create a user-friendly interface for medical professionals to input patient tumor characteristics and receive cancer risk predictions.

```python
# Required libraries for TUI inference application
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, FloatPrompt, IntPrompt
from rich.progress import track
from rich.layout import Layout
from rich.text import Text
from rich import print as rprint
from rich.columns import Columns
import joblib
import numpy as np
import pandas as pd
import os
```

### 24. Application Architecture

#### 24.1 Main Application Class

```python
class BreastCancerPredictor:
    def __init__(self):
        self.console = Console()
        self.model = None
        self.feature_names = None
        self.model_metrics = None
        self.test_results = None
        self.load_model()

    def load_model(self):
        """Load trained model and metadata"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir)

            self.model = joblib.load(os.path.join(model_dir, 'breast_cancer_model.pkl'))
            self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
            self.model_metrics = joblib.load(os.path.join(model_dir, 'model_metrics.pkl'))

            # Try to load test results if available
            try:
                self.test_results = joblib.load(os.path.join(model_dir, 'test_results.pkl'))
            except FileNotFoundError:
                self.test_results = None

            return True
        except FileNotFoundError as e:
            self.console.print(f"[red]Error: Model files not found! {e}[/red]")
            return False
```

#### 24.2 Feature Input Specifications

```python
# Define feature input specifications for breast cancer features
FEATURE_SPECS = {
    'radius_mean': {
        'prompt': 'Mean radius of cell nuclei',
        'type': 'float',
        'min': 5.0, 'max': 30.0,
        'description': 'Mean distance from center to points on perimeter (μm)'
    },
    'texture_mean': {
        'prompt': 'Mean texture of cell nuclei',
        'type': 'float',
        'min': 5.0, 'max': 40.0,
        'description': 'Standard deviation of gray-scale values'
    },
    'perimeter_mean': {
        'prompt': 'Mean perimeter of cell nuclei',
        'type': 'float',
        'min': 40.0, 'max': 200.0,
        'description': 'Mean perimeter of cell nuclei (μm)'
    },
    'area_mean': {
        'prompt': 'Mean area of cell nuclei',
        'type': 'float',
        'min': 100.0, 'max': 2500.0,
        'description': 'Mean area of cell nuclei (μm²)'
    },
    'smoothness_mean': {
        'prompt': 'Mean smoothness of cell nuclei',
        'type': 'float',
        'min': 0.05, 'max': 0.20,
        'description': 'Local variation in radius lengths'
    },
    'compactness_mean': {
        'prompt': 'Mean compactness of cell nuclei',
        'type': 'float',
        'min': 0.01, 'max': 0.35,
        'description': 'Perimeter² / area - 1.0'
    },
    'concavity_mean': {
        'prompt': 'Mean concavity of cell nuclei',
        'type': 'float',
        'min': 0.0, 'max': 0.45,
        'description': 'Severity of concave portions of contour'
    },
    'concave_points_mean': {
        'prompt': 'Mean concave points of cell nuclei',
        'type': 'float',
        'min': 0.0, 'max': 0.20,
        'description': 'Number of concave portions of contour'
    },
    'symmetry_mean': {
        'prompt': 'Mean symmetry of cell nuclei',
        'type': 'float',
        'min': 0.10, 'max': 0.30,
        'description': 'Symmetry of cell nuclei'
    },
    'fractal_dimension_mean': {
        'prompt': 'Mean fractal dimension of cell nuclei',
        'type': 'float',
        'min': 0.04, 'max': 0.10,
        'description': 'Coastline approximation - 1'
    }
}

# For this demo, we'll focus on the 10 mean features
# In practice, you could include all 30 features or select the most important ones
```

### 25. User Interface Implementation

#### 25.1 Welcome Screen and Model Information

```python
def display_welcome(self):
    """Display welcome screen with model information"""
    if self.test_results:
        accuracy = self.test_results['test_metrics']['accuracy']
        sensitivity = self.test_results['test_metrics']['sensitivity']
        specificity = self.test_results['test_metrics']['specificity']
        auc = self.test_results['test_metrics']['auc_roc']

        performance_text = (f"Model Performance (Test Set):\n"
                          f"• Accuracy: {accuracy:.1%}\n"
                          f"• Sensitivity: {sensitivity:.1%}\n"
                          f"• Specificity: {specificity:.1%}\n"
                          f"• AUC-ROC: {auc:.3f}")
    else:
        cv_auc = self.model_metrics.get('cv_auc_mean', 0)
        performance_text = f"Model Performance (CV): AUC = {cv_auc:.3f}"

    welcome_panel = Panel.fit(
        "[bold blue]Breast Cancer Prediction System[/bold blue]\n"
        "[dim]AI-Powered Clinical Decision Support Tool[/dim]\n\n"
        f"{performance_text}",
        title="🏥 Welcome",
        border_style="blue"
    )
    self.console.print(welcome_panel)

def display_disclaimer(self):
    """Display medical disclaimer"""
    disclaimer = Panel(
        "[yellow]⚠️  MEDICAL DISCLAIMER ⚠️[/yellow]\n\n"
        "This AI model is intended for educational and research purposes only.\n"
        "It should NOT be used as a substitute for professional medical diagnosis\n"
        "or pathological examination. Breast cancer diagnosis requires proper\n"
        "histopathological analysis by qualified pathologists.\n\n"
        "Always consult qualified healthcare providers for medical decisions.",
        border_style="yellow"
    )
    self.console.print(disclaimer)

    proceed = Confirm.ask("Do you acknowledge this disclaimer and wish to continue?")
    return proceed
```

#### 25.2 Interactive Data Collection

```python
def collect_patient_data(self):
    """Collect tumor characteristics through interactive prompts"""
    self.console.print("\n[bold green]Tumor Characteristics Collection[/bold green]")
    self.console.print("Please provide the following tumor measurements:\n")

    patient_data = {}

    # For this demo, collect only the 10 mean features
    # In practice, you might want to collect more features or the most important ones
    for feature, spec in track(FEATURE_SPECS.items(),
                              description="Collecting tumor data..."):

        # Display feature information
        info_panel = Panel(
            f"[bold]{spec['prompt']}[/bold]\n{spec['description']}\n"
            f"[dim]Valid range: {spec['min']:.2f} - {spec['max']:.2f}[/dim]",
            border_style="cyan"
        )
        self.console.print(info_panel)

        # Get user input
        value = FloatPrompt.ask(
            f"Enter {spec['prompt'].lower()}",
            default=None,
            show_default=False
        )

        # Validate range
        while value < spec['min'] or value > spec['max']:
            self.console.print(f"[red]Please enter a value between {spec['min']:.2f} and {spec['max']:.2f}[/red]")
            value = FloatPrompt.ask(f"Enter {spec['prompt'].lower()}")

        patient_data[feature] = value
        self.console.print(f"✓ {spec['prompt']}: [green]{value:.3f}[/green]\n")

    # For missing features, use dataset median values or ask user to specify
    # This is a simplified approach - in practice, you'd collect all features
    self.console.print("[yellow]Note: Using dataset averages for remaining features not collected.[/yellow]")

    return patient_data

def fill_missing_features(self, patient_data):
    """Fill missing features with dataset medians or reasonable defaults"""
    # This is a simplified approach - in practice, you'd want to collect all features
    # or use a more sophisticated imputation strategy

    default_values = {
        # SE features (standard errors) - typically smaller values
        'radius_se': 0.4, 'texture_se': 1.2, 'perimeter_se': 2.9, 'area_se': 40.0,
        'smoothness_se': 0.007, 'compactness_se': 0.025, 'concavity_se': 0.031,
        'concave_points_se': 0.012, 'symmetry_se': 0.020, 'fractal_dimension_se': 0.004,

        # Worst features (typically larger than mean values)
        'radius_worst': 16.3, 'texture_worst': 25.7, 'perimeter_worst': 107.3,
        'area_worst': 880.6, 'smoothness_worst': 0.132, 'compactness_worst': 0.254,
        'concavity_worst': 0.272, 'concave_points_worst': 0.115, 'symmetry_worst': 0.290,
        'fractal_dimension_worst': 0.084
    }

    complete_data = patient_data.copy()

    # Add missing features
    for feature in self.feature_names:
        if feature not in complete_data:
            if feature in default_values:
                complete_data[feature] = default_values[feature]
            else:
                # Estimate based on mean features if possible
                base_feature = feature.replace('_se', '').replace('_worst', '')
                if base_feature in complete_data:
                    if '_se' in feature:
                        complete_data[feature] = complete_data[base_feature] * 0.1  # SE ~10% of mean
                    elif '_worst' in feature:
                        complete_data[feature] = complete_data[base_feature] * 1.3  # Worst ~30% larger
                else:
                    complete_data[feature] = 0.0  # Last resort

    return complete_data
```

### 26. Model Inference and Risk Assessment

#### 26.1 Risk Prediction Logic

```python
def predict_risk(self, patient_data):
    """Make prediction and calculate risk assessment"""
    # Fill missing features
    complete_data = self.fill_missing_features(patient_data)

    # Convert to DataFrame with correct feature order
    input_df = pd.DataFrame([complete_data])[self.feature_names]

    # Get prediction and probability
    prediction = self.model.predict(input_df)[0]
    probability = self.model.predict_proba(input_df)[0, 1]

    return prediction, probability, complete_data

def interpret_risk_level(self, probability):
    """Interpret probability as risk level with clinical context"""
    if probability < 0.1:
        return {
            'level': 'Very Low',
            'color': 'green',
            'icon': '🟢',
            'message': 'Very low probability of malignancy. Tumor characteristics suggest benign nature.',
            'recommendations': [
                'Continue routine follow-up care',
                'Monitor for any changes in tumor characteristics',
                'Consider standard imaging surveillance protocol',
                'Pathological examination recommended for confirmation'
            ]
        }
    elif probability < 0.3:
        return {
            'level': 'Low',
            'color': 'blue',
            'icon': '🔵',
            'message': 'Low probability of malignancy. Features generally consistent with benign tumor.',
            'recommendations': [
                'Histopathological examination recommended',
                'Consider additional imaging if clinically indicated',
                'Short-term follow-up may be appropriate',
                'Discuss findings with pathology team'
            ]
        }
    elif probability < 0.6:
        return {
            'level': 'Moderate',
            'color': 'yellow',
            'icon': '🟡',
            'message': 'Moderate risk of malignancy. Further evaluation strongly recommended.',
            'recommendations': [
                'Immediate histopathological examination required',
                'Consider core needle biopsy if not already performed',
                'Multidisciplinary team consultation advised',
                'Additional imaging studies may be warranted'
            ]
        }
    elif probability < 0.8:
        return {
            'level': 'High',
            'color': 'orange',
            'icon': '🟠',
            'message': 'High probability of malignancy. Urgent pathological evaluation needed.',
            'recommendations': [
                'URGENT: Complete pathological workup required',
                'Multidisciplinary oncology team consultation',
                'Staging studies if malignancy confirmed',
                'Patient counseling and support services'
            ]
        }
    else:
        return {
            'level': 'Very High',
            'color': 'red',
            'icon': '🔴',
            'message': 'Very high probability of malignancy. Immediate comprehensive evaluation required.',
            'recommendations': [
                'IMMEDIATE: Complete oncological evaluation',
                'Rapid pathological diagnosis and staging',
                'Urgent multidisciplinary tumor board review',
                'Prepare for potential treatment planning'
            ]
        }
```

#### 26.2 Feature Contribution Analysis

```python
def analyze_risk_factors(self, complete_data, probability):
    """Analyze individual feature contributions to prediction"""

    # Get feature coefficients from the model
    classifier = self.model.named_steps['classifier']
    coefficients = classifier.coef_[0]

    # Calculate feature contributions
    scaler = self.model.named_steps['scaler']
    scaled_data = scaler.transform(pd.DataFrame([complete_data])[self.feature_names])

    malignancy_factors = []
    benign_factors = []

    for i, feature in enumerate(self.feature_names):
        contribution = scaled_data[0][i] * coefficients[i]

        factor_info = {
            'feature': feature,
            'value': complete_data[feature],
            'contribution': contribution,
            'coefficient': coefficients[i]
        }

        if contribution > 0.1:  # Significant malignancy contribution
            malignancy_factors.append(factor_info)
        elif contribution < -0.1:  # Significant benign contribution
            benign_factors.append(factor_info)

    # Sort by absolute contribution
    malignancy_factors.sort(key=lambda x: x['contribution'], reverse=True)
    benign_factors.sort(key=lambda x: x['contribution'])

    return malignancy_factors, benign_factors

def get_feature_description(self, feature, value):
    """Get clinical description for specific feature values"""

    feature_descriptions = {
        'radius_mean': f'{value:.2f} μm mean nuclear radius',
        'texture_mean': f'{value:.2f} texture variation',
        'perimeter_mean': f'{value:.2f} μm mean nuclear perimeter',
        'area_mean': f'{value:.1f} μm² mean nuclear area',
        'smoothness_mean': f'{value:.3f} smoothness index',
        'compactness_mean': f'{value:.3f} compactness measure',
        'concavity_mean': f'{value:.3f} concavity severity',
        'concave_points_mean': f'{value:.3f} concave points ratio',
        'symmetry_mean': f'{value:.3f} symmetry measure',
        'fractal_dimension_mean': f'{value:.3f} fractal dimension'
    }

    # Add clinical interpretation
    base_desc = feature_descriptions.get(feature, f'{value:.3f}')

    # Add range interpretation
    if 'radius' in feature or 'perimeter' in feature or 'area' in feature:
        if 'mean' in feature:
            if value > 15:
                base_desc += ' (enlarged)'
            elif value < 10:
                base_desc += ' (small)'
    elif 'texture' in feature:
        if value > 25:
            base_desc += ' (high variation)'
        elif value < 15:
            base_desc += ' (uniform)'

    return base_desc
```

### 27. Results Display and Clinical Interpretation

#### 27.1 Comprehensive Results Dashboard

```python
def display_results(self, patient_data, prediction, probability, risk_interpretation,
                   malignancy_factors, benign_factors, complete_data):
    """Display comprehensive results dashboard"""

    # Main prediction panel
    prediction_panel = Panel(
        f"{risk_interpretation['icon']} [bold {risk_interpretation['color']}]"
        f"{risk_interpretation['level']} Risk of Malignancy[/bold {risk_interpretation['color']}]\n\n"
        f"Probability: [bold]{probability:.1%}[/bold]\n"
        f"Prediction: [bold]{'Malignant' if prediction == 1 else 'Benign'}[/bold]\n\n"
        f"[italic]{risk_interpretation['message']}[/italic]",
        title="🔬 Tumor Analysis Results",
        border_style=risk_interpretation['color']
    )
    self.console.print(prediction_panel)

    # Malignancy risk factors
    if malignancy_factors:
        risk_table = Table(title="⚠️ Primary Malignancy Indicators", border_style="red")
        risk_table.add_column("Feature", style="bold")
        risk_table.add_column("Value", justify="center")
        risk_table.add_column("Clinical Impact", justify="center")

        for factor in malignancy_factors[:5]:  # Top 5 risk factors
            impact_desc = self.get_feature_description(factor['feature'], factor['value'])
            risk_table.add_row(
                factor['feature'].replace('_', ' ').title(),
                f"{factor['value']:.3f}",
                f"High Risk ({factor['contribution']:+.2f})"
            )

        self.console.print(risk_table)

    # Benign indicators
    if benign_factors:
        benign_table = Table(title="✅ Benign Characteristics", border_style="green")
        benign_table.add_column("Feature", style="bold")
        benign_table.add_column("Value", justify="center")
        benign_table.add_column("Clinical Impact", justify="center")

        for factor in benign_factors[:3]:  # Top 3 benign factors
            benign_table.add_row(
                factor['feature'].replace('_', ' ').title(),
                f"{factor['value']:.3f}",
                f"Benign ({factor['contribution']:+.2f})"
            )

        self.console.print(benign_table)

    # Clinical recommendations
    recommendations_panel = Panel(
        "\n".join([f"• {rec}" for rec in risk_interpretation['recommendations']]),
        title="🏥 Clinical Recommendations",
        border_style="blue"
    )
    self.console.print(recommendations_panel)

def display_patient_summary(self, patient_data):
    """Display patient tumor data summary for confirmation"""
    summary_table = Table(title="Tumor Characteristics Summary", border_style="cyan")
    summary_table.add_column("Parameter", style="bold")
    summary_table.add_column("Value", justify="center")
    summary_table.add_column("Description")

    for feature, value in patient_data.items():
        if feature in FEATURE_SPECS:
            spec = FEATURE_SPECS[feature]
            description = self.get_feature_description(feature, value)
            summary_table.add_row(
                spec['prompt'],
                f"{value:.3f}",
                description
            )

    self.console.print(summary_table)
```

### 28. Main Application Flow

#### 28.1 Complete Application Runner

```python
def run_application(self):
    """Main application flow"""
    self.console.clear()

    # Welcome and disclaimer
    self.display_welcome()
    if not self.display_disclaimer():
        self.console.print("[yellow]Application terminated by user.[/yellow]")
        return

    while True:
        try:
            # Collect tumor characteristics
            patient_data = self.collect_patient_data()

            # Confirm data
            self.display_patient_summary(patient_data)
            if not Confirm.ask("Are these tumor characteristics correct?"):
                continue

            # Make prediction
            with self.console.status("[bold green]Analyzing tumor characteristics..."):
                prediction, probability, complete_data = self.predict_risk(patient_data)
                risk_interpretation = self.interpret_risk_level(probability)
                malignancy_factors, benign_factors = self.analyze_risk_factors(
                    complete_data, probability)

            # Display results
            self.console.clear()
            self.display_results(patient_data, prediction, probability,
                               risk_interpretation, malignancy_factors, benign_factors,
                               complete_data)

            # Ask for another analysis
            if not Confirm.ask("\nWould you like to analyze another tumor sample?"):
                break

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Application interrupted by user.[/yellow]")
            break
        except Exception as e:
            self.console.print(f"[red]Error occurred: {str(e)}[/red]")
            if not Confirm.ask("Would you like to try again?"):
                break

    self.console.print("\n[blue]Thank you for using the Breast Cancer Prediction System![/blue]")
```

### 29. Application Entry Point and Usage

#### 29.1 Main Script Structure

```python
if __name__ == "__main__":
    app = BreastCancerPredictor()

    if app.model is None:
        console = Console()
        console.print("[red]Failed to load model. Please ensure model files exist.[/red]")
        console.print("Run cancer_training.py first to train the model.")
        exit(1)

    app.run_application()
```

## TUI Application Features Summary

The breast cancer prediction TUI application provides:

### User Experience Features

- **Professional Medical Interface:** Rich library for clinical-grade presentation
- **Guided Data Collection:** Step-by-step tumor characteristic input with validation
- **Medical Disclaimer:** Appropriate warnings for clinical AI applications
- **Data Validation:** Range checking for all tumor measurements
- **Confirmation Steps:** Review tumor data before analysis

### Clinical Intelligence Features

- **Risk Level Classification:** 5-tier malignancy risk assessment
- **Feature Contribution Analysis:** Identifies key tumor characteristics driving prediction
- **Clinical Recommendations:** Actionable next steps based on risk level
- **Pathology-Focused:** Appropriate for cytological/histological analysis workflow

### Technical Features

- **Robust Error Handling:** Graceful handling of missing or invalid data
- **Model Integration:** Seamless integration with trained scikit-learn pipeline
- **Feature Imputation:** Handles partially specified tumor characteristics
- **Professional Presentation:** Suitable for clinical decision support

This comprehensive plan creates a complete breast cancer prediction system from training through deployment, specifically adapted for the unique characteristics of cytological tumor analysis and clinical oncology workflows.
