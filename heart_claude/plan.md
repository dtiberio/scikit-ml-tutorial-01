# Heart Disease Prediction ML Pipeline - Training Plan

This tutorial outlines a comprehensive machine learning pipeline for predicting heart disease using Logistic Regression on the UCI Heart Disease dataset.

## Dataset Overview

The heart.csv dataset contains 303 records with 14 features:
- **Demographics:** age, sex
- **Clinical measurements:** chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs), resting ECG (restecg), max heart rate (thalach)
- **Exercise metrics:** exercise-induced angina (exang), ST depression (oldpeak), slope of peak exercise ST segment (slope)
- **Cardiac tests:** number of major vessels colored by fluoroscopy (ca), thalassemia type (thal)
- **Target:** heart disease presence (0=no disease, 1=disease)

## Step-by-Step ML Pipeline

### 1. Import Libraries and Load Data

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('heart.csv')
```

### 2. Exploratory Data Analysis (EDA)

**Purpose:** Understand data structure, distributions, and relationships before preprocessing.

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
print("\nTarget distribution:")
print(df['target'].value_counts())
print("Target proportion:")
print(df['target'].value_counts(normalize=True))
```

**Visualization:**
- Histograms for continuous variables
- Bar plots for categorical variables
- Correlation heatmap
- Box plots to identify outliers
- Target distribution analysis

### 3. Data Preprocessing

#### 3.1 Feature Engineering
```python
# Create additional features if needed
# Example: Age groups, BMI categories, etc.
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 70, 100], 
                        labels=['Young', 'Middle', 'Senior', 'Elderly'])
```

#### 3.2 Handle Categorical Variables
```python
# Identify categorical and numerical features
categorical_features = ['cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# One-hot encoding for categorical variables (if needed)
# Note: Many features are already numerically encoded
```

#### 3.3 Check for Outliers
```python
# Detect outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Check outliers in numerical features
for col in numerical_features:
    outliers = detect_outliers_iqr(df, col)
    print(f"{col}: {len(outliers)} outliers")
```

### 4. Feature Selection and Target Separation

```python
# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Feature names:", X.columns.tolist())
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
```

### 6. Feature Scaling

**Why:** Logistic Regression is sensitive to feature scales, so standardization is crucial.

```python
# Initialize scaler
scaler = StandardScaler()

# Fit on training data only to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully")
print("Training features mean:", np.mean(X_train_scaled, axis=0))
print("Training features std:", np.std(X_train_scaled, axis=0))
```

### 7. Model Training

#### 7.1 Initial Model Training
```python
# Initialize Logistic Regression model
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000  # Increase if convergence issues
)

# Train the model
lr_model.fit(X_train_scaled, y_train)
print("Model trained successfully!")
```

#### 7.2 Feature Importance Analysis
```python
# Get feature coefficients (importance)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("Top 10 most important features:")
print(feature_importance.head(10))
```

### 8. Model Evaluation on Training Data

#### 8.1 Cross-Validation
```python
# Perform cross-validation
cv_scores = cross_val_score(
    lr_model, X_train_scaled, y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
print("CV accuracy std:", cv_scores.std())
```

#### 8.2 Training Performance
```python
# Predictions on training set
y_train_pred = lr_model.predict(X_train_scaled)
y_train_prob = lr_model.predict_proba(X_train_scaled)[:, 1]

# Classification report
print("Training Set Classification Report:")
print(classification_report(y_train, y_train_pred))

# Confusion matrix
train_cm = confusion_matrix(y_train, y_train_pred)
print("Training Confusion Matrix:")
print(train_cm)

# AUC-ROC Score
train_auc = roc_auc_score(y_train, y_train_prob)
print(f"Training AUC-ROC Score: {train_auc:.4f}")
```

### 9. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'saga']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use the best model
best_lr_model = grid_search.best_estimator_
```

### 10. Pipeline Creation (Best Practice)

```python
# Create a complete pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Fit pipeline
pipeline.fit(X_train, y_train)
print("Pipeline trained successfully!")
```

### 11. Model Interpretation

```python
# Feature coefficients interpretation
coefficients = best_lr_model.coef_[0]
feature_names = X.columns

# Create interpretation dataframe
interpretation_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Odds_Ratio': np.exp(coefficients)
}).sort_values('Coefficient', ascending=False)

print("Feature Interpretation (Odds Ratios):")
print(interpretation_df)
```

### 12. Save the Trained Model

```python
import joblib

# Save the trained pipeline
joblib.dump(pipeline, 'heart_disease_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print("Model and scaler saved successfully!")
```

## Key Considerations for Logistic Regression

1. **Feature Scaling:** Critical for optimal performance
2. **Multicollinearity:** Check correlation between features
3. **Regularization:** Use L1/L2 penalties to prevent overfitting
4. **Class Imbalance:** Monitor if classes are balanced
5. **Convergence:** Increase max_iter if model doesn't converge
6. **Interpretability:** Coefficients represent log-odds ratios

## Expected Outputs

After completing this pipeline, you should have:
- A trained and tuned Logistic Regression model
- Feature importance rankings
- Cross-validation performance metrics
- A complete preprocessing pipeline
- Saved model files for future inference

## Next Steps: Model Testing and Evaluation Plan

This training plan prepares the foundation for comprehensive model testing. The following sections detail how to evaluate the trained model on test data.

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
pipeline = joblib.load('heart_disease_model.pkl')
feature_names = joblib.load('feature_names.pkl')
model_metrics = joblib.load('model_metrics.pkl')

print("Model loaded successfully!")
print("Best training parameters:", model_metrics['best_params'])
print("Training CV AUC:", model_metrics['cv_auc_mean'])
```

### 14. Test Set Predictions

**Purpose:** Generate predictions and probabilities on the held-out test set.

```python
# Make predictions on test set
y_test_pred = pipeline.predict(X_test)
y_test_prob = pipeline.predict_proba(X_test)[:, 1]

print("Test predictions generated")
print("Prediction distribution:", np.bincount(y_test_pred))
print("Probability range:", f"{y_test_prob.min():.3f} - {y_test_prob.max():.3f}")
```

### 15. Comprehensive Test Set Evaluation

#### 15.1 Classification Metrics
```python
# Calculate all classification metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)

print("=== TEST SET PERFORMANCE ===")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"AUC-ROC:   {test_auc:.4f}")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred, 
                          target_names=['No Disease', 'Disease']))
```

#### 15.2 Confusion Matrix Analysis
```python
# Generate and visualize confusion matrix
test_cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(test_cm)

# Calculate additional metrics from confusion matrix
tn, fp, fn, tp = test_cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
ppv = tp / (tp + fp)  # Positive Predictive Value
npv = tn / (tn + fn)  # Negative Predictive Value

print(f"\nAdditional Metrics:")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity:          {specificity:.4f}")
print(f"Positive Pred. Value: {ppv:.4f}")
print(f"Negative Pred. Value: {npv:.4f}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Test Set Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 16. ROC Curve and Precision-Recall Analysis

#### 16.1 ROC Curve
```python
# Generate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

plt.figure(figsize=(12, 5))

# ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {test_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Set')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
```

#### 16.2 Precision-Recall Curve
```python
# Generate Precision-Recall curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_prob)
pr_auc = auc(recall, precision)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2,
         label=f'PR curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Test Set')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('test_roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 17. Threshold Analysis and Optimization

**Purpose:** Find optimal prediction threshold for different objectives.

```python
# Calculate metrics for different thresholds
thresholds_analysis = []
for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_thresh = (y_test_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    thresholds_analysis.append({
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    })

threshold_df = pd.DataFrame(thresholds_analysis)

# Find optimal thresholds
opt_f1_thresh = threshold_df.loc[threshold_df['f1_score'].idxmax(), 'threshold']
opt_acc_thresh = threshold_df.loc[threshold_df['accuracy'].idxmax(), 'threshold']

print("Optimal Thresholds:")
print(f"Best F1-Score threshold: {opt_f1_thresh:.2f}")
print(f"Best Accuracy threshold: {opt_acc_thresh:.2f}")
print(f"Default threshold (0.5) metrics:")
print(threshold_df[threshold_df['threshold'] == 0.5])
```

### 18. Feature Importance Validation on Test Set

**Purpose:** Verify that important features remain consistent.

```python
# Get feature coefficients from trained model
trained_classifier = pipeline.named_steps['classifier']
test_coefficients = trained_classifier.coef_[0]

# Create feature importance dataframe
test_feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': test_coefficients,
    'Abs_Coefficient': np.abs(test_coefficients),
    'Odds_Ratio': np.exp(test_coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("Feature Importance (Test Model):")
print(test_feature_importance)

# Compare with training feature importance
print("\nTop 5 Most Important Features:")
for i, (_, row) in enumerate(test_feature_importance.head(5).iterrows()):
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"{i+1}. {row['Feature']}: {direction} heart disease risk "
          f"(coef: {row['Coefficient']:.3f}, OR: {row['Odds_Ratio']:.3f})")
```

### 19. Model Performance Comparison

**Purpose:** Compare training vs. test performance to check for overfitting.

```python
# Compare training and test metrics
comparison_metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Training_CV': [
        model_metrics.get('cv_accuracy_mean', 'N/A'),
        model_metrics.get('cv_precision_mean', 'N/A'),
        model_metrics.get('cv_recall_mean', 'N/A'),
        model_metrics.get('cv_f1_mean', 'N/A'),
        model_metrics['cv_auc_mean']
    ],
    'Test_Set': [test_accuracy, test_precision, test_recall, test_f1, test_auc]
}

comparison_df = pd.DataFrame(comparison_metrics)
print("=== TRAINING vs TEST PERFORMANCE ===")
print(comparison_df)

# Check for overfitting
auc_diff = model_metrics['cv_auc_mean'] - test_auc
if abs(auc_diff) < 0.05:
    print(f"\n✓ Good generalization (AUC diff: {auc_diff:.3f})")
elif auc_diff > 0.05:
    print(f"\n⚠ Possible overfitting (AUC diff: {auc_diff:.3f})")
else:
    print(f"\n⚠ Possible underfitting (AUC diff: {auc_diff:.3f})")
```

### 20. Error Analysis

**Purpose:** Analyze misclassified cases to understand model limitations.

```python
# Identify misclassified samples
misclassified_mask = y_test != y_test_pred
misclassified_indices = y_test.index[misclassified_mask]

print(f"Misclassified samples: {len(misclassified_indices)} out of {len(y_test)}")

# Analyze false positives and false negatives
false_positives = y_test.index[(y_test == 0) & (y_test_pred == 1)]
false_negatives = y_test.index[(y_test == 1) & (y_test_pred == 0)]

print(f"False Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")

# Examine characteristics of misclassified cases
if len(false_positives) > 0:
    print("\nFalse Positive characteristics:")
    fp_data = X_test.loc[false_positives]
    print(fp_data.describe())

if len(false_negatives) > 0:
    print("\nFalse Negative characteristics:")
    fn_data = X_test.loc[false_negatives]
    print(fn_data.describe())
```

### 21. Clinical Interpretation

**Purpose:** Interpret results from a medical perspective.

```python
print("=== CLINICAL INTERPRETATION ===")
print("\nModel Performance Summary:")
print(f"• The model correctly identifies {test_recall:.1%} of patients with heart disease")
print(f"• Of patients predicted to have disease, {test_precision:.1%} actually do")
print(f"• Overall accuracy of {test_accuracy:.1%} on unseen patients")
print(f"• AUC of {test_auc:.3f} indicates {'excellent' if test_auc > 0.9 else 'good' if test_auc > 0.8 else 'fair'} discrimination")

print(f"\nClinical Impact:")
print(f"• False Negatives ({len(false_negatives)}): Patients with disease missed by model")
print(f"• False Positives ({len(false_positives)}): Healthy patients incorrectly flagged")

print(f"\nKey Risk Factors (Top 3):")
for i, (_, row) in enumerate(test_feature_importance.head(3).iterrows()):
    impact = "High risk" if row['Coefficient'] > 0 else "Protective"
    print(f"  {i+1}. {row['Feature']}: {impact} factor (OR: {row['Odds_Ratio']:.2f})")
```

### 22. Save Test Results

**Purpose:** Persist test results for reporting and future reference.

```python
# Save test results
test_results = {
    'test_metrics': {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'auc_roc': test_auc,
        'specificity': specificity,
        'sensitivity': sensitivity
    },
    'confusion_matrix': test_cm.tolist(),
    'optimal_thresholds': {
        'best_f1': opt_f1_thresh,
        'best_accuracy': opt_acc_thresh
    },
    'feature_importance': test_feature_importance.to_dict('records'),
    'misclassification_analysis': {
        'total_misclassified': len(misclassified_indices),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives)
    }
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
- [ ] Test set performance is reasonable (not overfitting/underfitting)
- [ ] Confusion matrix and classification metrics calculated
- [ ] ROC and Precision-Recall curves generated
- [ ] Optimal thresholds identified for different objectives
- [ ] Feature importance validated and interpreted
- [ ] Error analysis completed for misclassified cases
- [ ] Clinical interpretation provided
- [ ] All results and visualizations saved

## Key Testing Considerations

1. **Generalization:** Test performance should be close to CV performance
2. **Clinical Relevance:** Consider the cost of false negatives vs. false positives
3. **Threshold Selection:** Choose based on clinical objectives (sensitivity vs. specificity)
4. **Feature Stability:** Important features should remain consistent
5. **Error Analysis:** Understand what types of cases the model struggles with

This comprehensive testing plan ensures thorough evaluation of the trained heart disease prediction model before deployment or clinical use.

---

## Model Inference Plan - Interactive TUI Application

After training and testing the model, the final step is creating an interactive application for real-world inference. This section details building a Text User Interface (TUI) using Python's `rich` library for professional presentation and user interaction.

### 23. Interactive Inference Application Setup

**Purpose:** Create a user-friendly interface for medical professionals to input patient data and receive predictions.

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
```

### 24. Application Architecture

#### 24.1 Main Application Class
```python
class HeartDiseasePredictor:
    def __init__(self):
        self.console = Console()
        self.model = None
        self.feature_names = None
        self.model_metrics = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and metadata"""
        try:
            self.model = joblib.load('heart_disease_model.pkl')
            self.feature_names = joblib.load('feature_names.pkl')
            self.model_metrics = joblib.load('model_metrics.pkl')
            return True
        except FileNotFoundError:
            self.console.print("[red]Error: Model files not found![/red]")
            return False
```

#### 24.2 Feature Input Definitions
```python
# Define feature input specifications
FEATURE_SPECS = {
    'age': {
        'prompt': 'Patient age',
        'type': 'int',
        'min': 20, 'max': 100,
        'description': 'Age in years (20-100)'
    },
    'sex': {
        'prompt': 'Patient sex',
        'type': 'choice',
        'choices': {'Male': 1, 'Female': 0},
        'description': 'Biological sex'
    },
    'cp': {
        'prompt': 'Chest pain type',
        'type': 'choice',
        'choices': {
            'Typical Angina': 0,
            'Atypical Angina': 1, 
            'Non-Anginal Pain': 2,
            'Asymptomatic': 3
        },
        'description': 'Type of chest pain experienced'
    },
    'trestbps': {
        'prompt': 'Resting blood pressure',
        'type': 'int',
        'min': 80, 'max': 200,
        'description': 'Resting blood pressure in mm Hg'
    },
    'chol': {
        'prompt': 'Serum cholesterol',
        'type': 'int', 
        'min': 100, 'max': 600,
        'description': 'Serum cholesterol in mg/dl'
    },
    'fbs': {
        'prompt': 'Fasting blood sugar',
        'type': 'choice',
        'choices': {'> 120 mg/dl': 1, '≤ 120 mg/dl': 0},
        'description': 'Fasting blood sugar level'
    },
    'restecg': {
        'prompt': 'Resting ECG results',
        'type': 'choice',
        'choices': {
            'Normal': 0,
            'ST-T Wave Abnormality': 1,
            'Left Ventricular Hypertrophy': 2
        },
        'description': 'Resting electrocardiographic results'
    },
    'thalach': {
        'prompt': 'Maximum heart rate achieved',
        'type': 'int',
        'min': 60, 'max': 220,
        'description': 'Maximum heart rate achieved during exercise'
    },
    'exang': {
        'prompt': 'Exercise induced angina',
        'type': 'choice',
        'choices': {'Yes': 1, 'No': 0},
        'description': 'Exercise-induced angina'
    },
    'oldpeak': {
        'prompt': 'ST depression',
        'type': 'float',
        'min': 0.0, 'max': 10.0,
        'description': 'ST depression induced by exercise relative to rest'
    },
    'slope': {
        'prompt': 'Slope of peak exercise ST segment',
        'type': 'choice',
        'choices': {
            'Upsloping': 0,
            'Flat': 1,
            'Downsloping': 2
        },
        'description': 'Slope of the peak exercise ST segment'
    },
    'ca': {
        'prompt': 'Number of major vessels',
        'type': 'choice',
        'choices': {'0': 0, '1': 1, '2': 2, '3': 3},
        'description': 'Number of major vessels colored by fluoroscopy'
    },
    'thal': {
        'prompt': 'Thalassemia',
        'type': 'choice',
        'choices': {
            'Normal': 1,
            'Fixed Defect': 2,
            'Reversible Defect': 3
        },
        'description': 'Thalassemia type'
    }
}
```

### 25. User Interface Implementation

#### 25.1 Welcome Screen and Model Information
```python
def display_welcome(self):
    """Display welcome screen with model information"""
    welcome_panel = Panel.fit(
        "[bold blue]Heart Disease Prediction System[/bold blue]\n"
        "[dim]AI-Powered Clinical Decision Support Tool[/dim]\n\n"
        f"Model Performance (Test Set):\n"
        f"• Accuracy: {self.model_metrics.get('test_accuracy', 'N/A'):.1%}\n"
        f"• Sensitivity: {self.model_metrics.get('test_sensitivity', 'N/A'):.1%}\n"
        f"• Specificity: {self.model_metrics.get('test_specificity', 'N/A'):.1%}",
        title="🏥 Welcome",
        border_style="blue"
    )
    self.console.print(welcome_panel)

def display_disclaimer(self):
    """Display medical disclaimer"""
    disclaimer = Panel(
        "[yellow]⚠️  MEDICAL DISCLAIMER ⚠️[/yellow]\n\n"
        "This AI model is intended for educational and research purposes only.\n"
        "It should NOT be used as a substitute for professional medical advice,\n"
        "diagnosis, or treatment. Always consult qualified healthcare providers\n"
        "for medical decisions.",
        border_style="yellow"
    )
    self.console.print(disclaimer)
    
    proceed = Confirm.ask("Do you acknowledge this disclaimer and wish to continue?")
    return proceed
```

#### 25.2 Interactive Data Collection
```python
def collect_patient_data(self):
    """Collect patient data through interactive prompts"""
    self.console.print("\n[bold green]Patient Data Collection[/bold green]")
    self.console.print("Please provide the following patient information:\n")
    
    patient_data = {}
    
    for feature, spec in track(FEATURE_SPECS.items(), 
                              description="Collecting patient data..."):
        
        # Display feature information
        info_panel = Panel(
            f"[bold]{spec['prompt']}[/bold]\n{spec['description']}",
            border_style="cyan"
        )
        self.console.print(info_panel)
        
        # Get user input based on feature type
        if spec['type'] == 'int':
            value = IntPrompt.ask(
                f"Enter {spec['prompt'].lower()}",
                default=None,
                show_default=False
            )
            # Validate range
            while value < spec['min'] or value > spec['max']:
                self.console.print(f"[red]Please enter a value between {spec['min']} and {spec['max']}[/red]")
                value = IntPrompt.ask(f"Enter {spec['prompt'].lower()}")
                
        elif spec['type'] == 'float':
            value = FloatPrompt.ask(
                f"Enter {spec['prompt'].lower()}",
                default=None,
                show_default=False
            )
            # Validate range
            while value < spec['min'] or value > spec['max']:
                self.console.print(f"[red]Please enter a value between {spec['min']} and {spec['max']}[/red]")
                value = FloatPrompt.ask(f"Enter {spec['prompt'].lower()}")
                
        elif spec['type'] == 'choice':
            self.console.print("Available options:")
            for option, code in spec['choices'].items():
                self.console.print(f"  [cyan]{option}[/cyan] ({code})")
            
            choice = Prompt.ask(
                f"Select {spec['prompt'].lower()}",
                choices=list(spec['choices'].keys())
            )
            value = spec['choices'][choice]
        
        patient_data[feature] = value
        self.console.print(f"✓ {spec['prompt']}: [green]{value}[/green]\n")
    
    return patient_data
```

### 26. Model Inference and Risk Assessment

#### 26.1 Risk Prediction Logic
```python
def predict_risk(self, patient_data):
    """Make prediction and calculate risk assessment"""
    # Convert to DataFrame with correct feature order
    input_df = pd.DataFrame([patient_data])[self.feature_names]
    
    # Get prediction and probability
    prediction = self.model.predict(input_df)[0]
    probability = self.model.predict_proba(input_df)[0, 1]
    
    return prediction, probability

def interpret_risk_level(self, probability):
    """Interpret probability as risk level with clinical context"""
    if probability < 0.2:
        return {
            'level': 'Very Low',
            'color': 'green',
            'icon': '🟢',
            'message': 'Very low probability of heart disease. Continue regular preventive care.',
            'recommendations': [
                'Maintain healthy lifestyle habits',
                'Regular exercise and balanced diet',
                'Annual check-ups with healthcare provider',
                'Monitor cardiovascular risk factors'
            ]
        }
    elif probability < 0.4:
        return {
            'level': 'Low',
            'color': 'blue',
            'icon': '🔵',
            'message': 'Low probability of heart disease. Maintain current health practices.',
            'recommendations': [
                'Continue healthy lifestyle',
                'Monitor blood pressure and cholesterol',
                'Consider lifestyle modifications if risk factors present',
                'Discuss with healthcare provider'
            ]
        }
    elif probability < 0.6:
        return {
            'level': 'Moderate',
            'color': 'yellow',
            'icon': '🟡',
            'message': 'Moderate risk of heart disease. Consider further evaluation.',
            'recommendations': [
                'Schedule comprehensive cardiovascular evaluation',
                'Lifestyle modifications may be beneficial',
                'Monitor blood pressure and cholesterol regularly',
                'Discuss risk factors with cardiologist'
            ]
        }
    elif probability < 0.8:
        return {
            'level': 'High',
            'color': 'orange',
            'icon': '🟠',
            'message': 'High probability of heart disease. Further evaluation recommended.',
            'recommendations': [
                'Urgent cardiology consultation recommended',
                'Comprehensive cardiac workup may be needed',
                'Immediate lifestyle modifications advised',
                'Consider stress testing or imaging studies'
            ]
        }
    else:
        return {
            'level': 'Very High',
            'color': 'red',
            'icon': '🔴',
            'message': 'Very high probability of heart disease. Immediate medical attention advised.',
            'recommendations': [
                'IMMEDIATE cardiology consultation',
                'Comprehensive cardiac evaluation urgently needed',
                'Consider emergency department if symptoms present',
                'Aggressive risk factor modification required'
            ]
        }
```

#### 26.2 Risk Factor Analysis
```python
def analyze_risk_factors(self, patient_data, probability):
    """Analyze individual risk factors and their contributions"""
    
    # Get feature coefficients from the model
    classifier = self.model.named_steps['classifier']
    coefficients = classifier.coef_[0]
    
    # Calculate feature contributions
    scaler = self.model.named_steps['scaler']
    scaled_data = scaler.transform(pd.DataFrame([patient_data])[self.feature_names])
    
    risk_factors = []
    protective_factors = []
    
    for i, feature in enumerate(self.feature_names):
        contribution = scaled_data[0][i] * coefficients[i]
        
        factor_info = {
            'feature': feature,
            'value': patient_data[feature],
            'contribution': contribution,
            'coefficient': coefficients[i]
        }
        
        if contribution > 0.1:  # Significant risk contribution
            risk_factors.append(factor_info)
        elif contribution < -0.1:  # Significant protective contribution
            protective_factors.append(factor_info)
    
    # Sort by absolute contribution
    risk_factors.sort(key=lambda x: x['contribution'], reverse=True)
    protective_factors.sort(key=lambda x: x['contribution'])
    
    return risk_factors, protective_factors
```

### 27. Results Display and Interpretation

#### 27.1 Comprehensive Results Dashboard
```python
def display_results(self, patient_data, prediction, probability, risk_interpretation, 
                   risk_factors, protective_factors):
    """Display comprehensive results dashboard"""
    
    # Main prediction panel
    prediction_panel = Panel(
        f"{risk_interpretation['icon']} [bold {risk_interpretation['color']}]"
        f"{risk_interpretation['level']} Risk[/bold {risk_interpretation['color']}]\n\n"
        f"Probability: [bold]{probability:.1%}[/bold]\n"
        f"Prediction: [bold]{'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'}[/bold]\n\n"
        f"[italic]{risk_interpretation['message']}[/italic]",
        title="🏥 Risk Assessment",
        border_style=risk_interpretation['color']
    )
    self.console.print(prediction_panel)
    
    # Risk factors analysis
    if risk_factors:
        risk_table = Table(title="⚠️ Primary Risk Factors", border_style="red")
        risk_table.add_column("Factor", style="bold")
        risk_table.add_column("Value", justify="center")
        risk_table.add_column("Impact", justify="center")
        
        for factor in risk_factors[:5]:  # Top 5 risk factors
            impact_desc = self.get_factor_description(factor['feature'], factor['value'])
            risk_table.add_row(
                factor['feature'].replace('_', ' ').title(),
                str(factor['value']),
                f"High Risk ({factor['contribution']:+.2f})"
            )
        
        self.console.print(risk_table)
    
    # Protective factors
    if protective_factors:
        protective_table = Table(title="✅ Protective Factors", border_style="green")
        protective_table.add_column("Factor", style="bold")
        protective_table.add_column("Value", justify="center")
        protective_table.add_column("Impact", justify="center")
        
        for factor in protective_factors[:3]:  # Top 3 protective factors
            protective_table.add_row(
                factor['feature'].replace('_', ' ').title(),
                str(factor['value']),
                f"Protective ({factor['contribution']:+.2f})"
            )
        
        self.console.print(protective_table)
    
    # Clinical recommendations
    recommendations_panel = Panel(
        "\n".join([f"• {rec}" for rec in risk_interpretation['recommendations']]),
        title="🏥 Clinical Recommendations",
        border_style="blue"
    )
    self.console.print(recommendations_panel)

def get_factor_description(self, feature, value):
    """Get clinical description for specific factor values"""
    descriptions = {
        'age': f"{value} years old" + (" (advanced age increases risk)" if value > 65 else ""),
        'sex': "Male" if value == 1 else "Female" + (" (males at higher risk)" if value == 1 else ""),
        'cp': {0: "Typical angina (high risk)", 1: "Atypical angina", 
               2: "Non-anginal pain", 3: "Asymptomatic"}[value],
        'trestbps': f"{value} mmHg" + (" (hypertensive)" if value > 140 else " (normal)"),
        'chol': f"{value} mg/dl" + (" (high cholesterol)" if value > 240 else " (acceptable)"),
        'fbs': "Diabetic (>120 mg/dl)" if value == 1 else "Non-diabetic",
        'restecg': {0: "Normal ECG", 1: "ST-T abnormality", 
                   2: "Left ventricular hypertrophy"}[value],
        'thalach': f"{value} bpm" + (" (low max heart rate)" if value < 120 else " (good)"),
        'exang': "Exercise-induced angina present" if value == 1 else "No exercise angina",
        'oldpeak': f"{value} ST depression" + (" (significant)" if value > 2 else ""),
        'slope': {0: "Upsloping (good)", 1: "Flat", 2: "Downsloping (concerning)"}[value],
        'ca': f"{value} major vessels blocked" + (" (significant blockage)" if value > 1 else ""),
        'thal': {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}[value]
    }
    return descriptions.get(feature, str(value))
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
            # Collect patient data
            patient_data = self.collect_patient_data()
            
            # Confirm data
            self.display_patient_summary(patient_data)
            if not Confirm.ask("Is this information correct?"):
                continue
            
            # Make prediction
            with self.console.status("[bold green]Analyzing patient data..."):
                prediction, probability = self.predict_risk(patient_data)
                risk_interpretation = self.interpret_risk_level(probability)
                risk_factors, protective_factors = self.analyze_risk_factors(
                    patient_data, probability)
            
            # Display results
            self.console.clear()
            self.display_results(patient_data, prediction, probability, 
                               risk_interpretation, risk_factors, protective_factors)
            
            # Ask for another prediction
            if not Confirm.ask("\nWould you like to analyze another patient?"):
                break
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Application interrupted by user.[/yellow]")
            break
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")
            if not Confirm.ask("Would you like to try again?"):
                break
    
    self.console.print("\n[blue]Thank you for using the Heart Disease Prediction System![/blue]")

def display_patient_summary(self, patient_data):
    """Display patient data summary for confirmation"""
    summary_table = Table(title="Patient Data Summary", border_style="cyan")
    summary_table.add_column("Parameter", style="bold")
    summary_table.add_column("Value", justify="center")
    summary_table.add_column("Description")
    
    for feature, value in patient_data.items():
        spec = FEATURE_SPECS[feature]
        description = self.get_factor_description(feature, value)
        summary_table.add_row(
            spec['prompt'],
            str(value),
            description
        )
    
    self.console.print(summary_table)
```

### 29. Application Entry Point and Usage

#### 29.1 Main Script Structure
```python
if __name__ == "__main__":
    app = HeartDiseasePredictor()
    
    if app.model is None:
        console = Console()
        console.print("[red]Failed to load model. Please ensure model files exist.[/red]")
        console.print("Run heart_training.py first to train the model.")
        exit(1)
    
    app.run_application()
```

## TUI Application Features Summary

The interactive inference application provides:

### User Experience Features
- **Professional TUI:** Rich library for colored, formatted output
- **Interactive Data Collection:** Step-by-step patient data input with validation
- **Medical Disclaimer:** Proper legal/ethical warnings
- **Data Validation:** Range checking and input validation
- **Confirmation Steps:** User can review and confirm data before prediction

### Clinical Intelligence Features
- **Risk Level Interpretation:** 5-tier risk classification with clinical context
- **Factor Analysis:** Identifies primary risk and protective factors
- **Clinical Recommendations:** Actionable next steps based on risk level
- **Comprehensive Dashboard:** Visual presentation of all results

### Technical Features
- **Error Handling:** Graceful handling of errors and interruptions
- **Model Integration:** Seamless integration with trained pipeline
- **Data Persistence:** All results properly formatted and interpretable
- **Professional Presentation:** Medical-grade interface suitable for clinical settings

This TUI application transforms the raw ML model into a practical clinical decision support tool that medical professionals can use for patient risk assessment.