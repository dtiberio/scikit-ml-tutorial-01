# Heart Disease Prediction ML Pipeline - Testing Script
# Following the comprehensive plan for model testing and evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score, auc)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("HEART DISEASE PREDICTION - TESTING PIPELINE")
print("=" * 60)

# =============================================================================
# 1. SETUP AND DATA LOADING
# =============================================================================
print("\n1. SETUP AND DATA LOADING...")
print("-" * 40)

# Get script and parent directories
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
print(f"Script directory: {script_dir}")
print(f"Parent directory: {parent_dir}")

# Load original data to recreate train/test split
df = pd.read_csv(os.path.join(parent_dir, 'heart.csv'))
print(f"Original dataset shape: {df.shape}")

# Recreate the exact same train/test split used in training
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Test set shape: {X_test.shape}")
print(f"Test target distribution: {y_test.value_counts().to_dict()}")

# =============================================================================
# 2. LOAD TRAINED MODEL AND METADATA
# =============================================================================
print("\n2. LOAD TRAINED MODEL AND METADATA...")
print("-" * 40)

try:
    # Load the trained pipeline and metadata
    pipeline = joblib.load(os.path.join(script_dir, 'heart_disease_model.pkl'))
    feature_names = joblib.load(os.path.join(script_dir, 'feature_names.pkl'))
    model_metrics = joblib.load(os.path.join(script_dir, 'model_metrics.pkl'))
    
    print("✓ Model loaded successfully!")
    print("✓ Feature names loaded successfully!")
    print("✓ Model metrics loaded successfully!")
    
    print(f"\nTraining Information:")
    print(f"Best parameters: {model_metrics['best_params']}")
    print(f"Training CV AUC: {model_metrics['cv_auc_mean']:.4f} ± {model_metrics['cv_auc_std']:.4f}")
    print(f"Number of features: {len(feature_names)}")
    
except FileNotFoundError as e:
    print(f"❌ Error loading model files: {e}")
    print("Please run heart_training.py first to train and save the model.")
    exit(1)

# =============================================================================
# 3. GENERATE TEST SET PREDICTIONS
# =============================================================================
print("\n3. GENERATE TEST SET PREDICTIONS...")
print("-" * 40)

# Make predictions on test set
y_test_pred = pipeline.predict(X_test)
y_test_prob = pipeline.predict_proba(X_test)[:, 1]

print("✓ Test predictions generated successfully!")
print(f"Prediction distribution: {dict(zip(['No Disease', 'Disease'], np.bincount(y_test_pred)))}")
print(f"Probability range: {y_test_prob.min():.3f} - {y_test_prob.max():.3f}")
print(f"Mean probability: {y_test_prob.mean():.3f}")

# =============================================================================
# 4. COMPREHENSIVE TEST SET EVALUATION
# =============================================================================
print("\n4. COMPREHENSIVE TEST SET EVALUATION...")
print("-" * 40)

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
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred, 
                          target_names=['No Disease', 'Disease']))

# =============================================================================
# 5. CONFUSION MATRIX ANALYSIS
# =============================================================================
print("\n5. CONFUSION MATRIX ANALYSIS...")
print("-" * 40)

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

print(f"\nAdditional Clinical Metrics:")
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

# Add text annotations with percentages
total = test_cm.sum()
for i in range(2):
    for j in range(2):
        plt.text(j+0.5, i+0.7, f'({test_cm[i,j]/total:.1%})', 
                ha='center', va='center', fontsize=10, color='red')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. ROC CURVE AND PRECISION-RECALL ANALYSIS
# =============================================================================
print("\n6. ROC AND PRECISION-RECALL ANALYSIS...")
print("-" * 40)

# Generate ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_prob)

# Generate Precision-Recall curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(15, 5))

# ROC Curve
plt.subplot(1, 3, 1)
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

# Precision-Recall Curve
plt.subplot(1, 3, 2)
plt.plot(recall, precision, color='blue', lw=2,
         label=f'PR curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Test Set')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)

# Probability Distribution
plt.subplot(1, 3, 3)
plt.hist(y_test_prob[y_test == 0], bins=20, alpha=0.5, 
         label='No Disease', color='lightblue')
plt.hist(y_test_prob[y_test == 1], bins=20, alpha=0.5, 
         label='Disease', color='lightcoral')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Probability Distribution by Class')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'test_roc_pr_curves.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ ROC AUC: {test_auc:.4f}")
print(f"✓ PR AUC: {pr_auc:.4f}")

# =============================================================================
# 7. THRESHOLD ANALYSIS AND OPTIMIZATION
# =============================================================================
print("\n7. THRESHOLD ANALYSIS AND OPTIMIZATION...")
print("-" * 40)

# Calculate metrics for different thresholds
thresholds_analysis = []
thresholds_range = np.arange(0.1, 0.95, 0.05)

for threshold in thresholds_range:
    y_pred_thresh = (y_test_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    # Calculate specificity
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    if cm_thresh.shape == (2, 2):
        tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
        spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
    else:
        spec = 0
    
    thresholds_analysis.append({
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'specificity': spec
    })

threshold_df = pd.DataFrame(thresholds_analysis)

# Find optimal thresholds
opt_f1_thresh = threshold_df.loc[threshold_df['f1_score'].idxmax(), 'threshold']
opt_acc_thresh = threshold_df.loc[threshold_df['accuracy'].idxmax(), 'threshold']

print("Optimal Thresholds:")
print(f"Best F1-Score threshold: {opt_f1_thresh:.2f}")
print(f"Best Accuracy threshold: {opt_acc_thresh:.2f}")

# Find threshold closest to 0.5
default_idx = np.argmin(np.abs(threshold_df['threshold'] - 0.5))
print(f"\nDefault threshold (0.5) metrics:")
default_metrics = threshold_df.iloc[default_idx]
for col in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
    print(f"  {col}: {default_metrics[col]:.4f}")

# Visualize threshold analysis
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(threshold_df['threshold'], threshold_df['accuracy'], 'o-', label='Accuracy')
plt.plot(threshold_df['threshold'], threshold_df['f1_score'], 's-', label='F1-Score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Accuracy and F1-Score vs Threshold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(threshold_df['threshold'], threshold_df['precision'], 'o-', label='Precision')
plt.plot(threshold_df['threshold'], threshold_df['recall'], 's-', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(threshold_df['threshold'], threshold_df['recall'], 'o-', label='Sensitivity')
plt.plot(threshold_df['threshold'], threshold_df['specificity'], 's-', label='Specificity')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Sensitivity and Specificity vs Threshold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 2, 4)
# Youden's J statistic (Sensitivity + Specificity - 1)
youden_j = threshold_df['recall'] + threshold_df['specificity'] - 1
opt_youden_thresh = threshold_df.loc[youden_j.idxmax(), 'threshold']
plt.plot(threshold_df['threshold'], youden_j, 'o-', color='purple', label="Youden's J")
plt.axvline(x=opt_youden_thresh, color='red', linestyle='--', 
           label=f'Optimal J={opt_youden_thresh:.2f}')
plt.xlabel('Threshold')
plt.ylabel("Youden's J Statistic")
plt.title("Youden's J vs Threshold")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"Optimal Youden's J threshold: {opt_youden_thresh:.2f}")

# =============================================================================
# 8. FEATURE IMPORTANCE VALIDATION
# =============================================================================
print("\n8. FEATURE IMPORTANCE VALIDATION...")
print("-" * 40)

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
print(f"\nTop 5 Most Important Features:")
for i, (_, row) in enumerate(test_feature_importance.head(5).iterrows()):
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  {i+1}. {row['Feature']}: {direction} heart disease risk "
          f"(coef: {row['Coefficient']:.3f}, OR: {row['Odds_Ratio']:.3f})")

# Visualize feature importance
plt.figure(figsize=(12, 8))
colors = ['red' if x < 0 else 'blue' for x in test_feature_importance['Coefficient']]
bars = plt.barh(range(len(test_feature_importance)), test_feature_importance['Coefficient'], 
                color=colors, alpha=0.7)

plt.yticks(range(len(test_feature_importance)), test_feature_importance['Feature'])
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Feature Coefficients (Test Model)')
plt.grid(axis='x', alpha=0.3)

# Add coefficient values on bars
for i, (bar, coef) in enumerate(zip(bars, test_feature_importance['Coefficient'])):
    plt.text(coef + (0.01 if coef > 0 else -0.01), i, f'{coef:.3f}', 
             va='center', ha='left' if coef > 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'test_feature_coefficients.png'), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 9. MODEL PERFORMANCE COMPARISON
# =============================================================================
print("\n9. MODEL PERFORMANCE COMPARISON...")
print("-" * 40)

# Compare training and test metrics
comparison_metrics = {
    'Metric': ['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Training_CV': [
        model_metrics['cv_auc_mean'],
        'N/A',  # We didn't save CV accuracy in training
        'N/A',  # We didn't save CV precision in training
        'N/A',  # We didn't save CV recall in training
        'N/A'   # We didn't save CV F1 in training
    ],
    'Test_Set': [test_auc, test_accuracy, test_precision, test_recall, test_f1]
}

comparison_df = pd.DataFrame(comparison_metrics)
print("=== TRAINING vs TEST PERFORMANCE ===")
print(comparison_df)

# Check for overfitting based on AUC (the metric we have for both)
auc_diff = model_metrics['cv_auc_mean'] - test_auc
print(f"\nGeneralization Analysis:")
print(f"Training CV AUC: {model_metrics['cv_auc_mean']:.4f} ± {model_metrics['cv_auc_std']:.4f}")
print(f"Test AUC:        {test_auc:.4f}")
print(f"Difference:      {auc_diff:.4f}")

if abs(auc_diff) < 0.05:
    print(f"✓ Good generalization (AUC diff: {auc_diff:.3f})")
elif auc_diff > 0.05:
    print(f"⚠ Possible overfitting (AUC diff: {auc_diff:.3f})")
else:
    print(f"⚠ Possible underfitting (AUC diff: {auc_diff:.3f})")

# =============================================================================
# 10. ERROR ANALYSIS
# =============================================================================
print("\n10. ERROR ANALYSIS...")
print("-" * 40)

# Identify misclassified samples
misclassified_mask = y_test != y_test_pred
misclassified_indices = y_test.index[misclassified_mask]

print(f"Misclassified samples: {len(misclassified_indices)} out of {len(y_test)}")
print(f"Classification accuracy: {(len(y_test) - len(misclassified_indices))/len(y_test):.1%}")

# Analyze false positives and false negatives
false_positives = y_test.index[(y_test == 0) & (y_test_pred == 1)]
false_negatives = y_test.index[(y_test == 1) & (y_test_pred == 0)]

print(f"\nError Breakdown:")
print(f"False Positives: {len(false_positives)} (healthy patients predicted as diseased)")
print(f"False Negatives: {len(false_negatives)} (diseased patients predicted as healthy)")

# Examine characteristics of misclassified cases
if len(false_positives) > 0:
    print(f"\nFalse Positive Analysis:")
    fp_data = X_test.loc[false_positives]
    fp_probs = y_test_prob[y_test.index.isin(false_positives)]
    print(f"Average prediction probability: {fp_probs.mean():.3f}")
    print(f"Probability range: {fp_probs.min():.3f} - {fp_probs.max():.3f}")
    
    print("Key characteristics (mean values):")
    for feature in test_feature_importance.head(5)['Feature']:
        mean_fp = fp_data[feature].mean()
        mean_all = X_test[feature].mean()
        print(f"  {feature}: {mean_fp:.2f} (overall: {mean_all:.2f})")

if len(false_negatives) > 0:
    print(f"\nFalse Negative Analysis:")
    fn_data = X_test.loc[false_negatives]
    fn_probs = y_test_prob[y_test.index.isin(false_negatives)]
    print(f"Average prediction probability: {fn_probs.mean():.3f}")
    print(f"Probability range: {fn_probs.min():.3f} - {fn_probs.max():.3f}")
    
    print("Key characteristics (mean values):")
    for feature in test_feature_importance.head(5)['Feature']:
        mean_fn = fn_data[feature].mean()
        mean_all = X_test[feature].mean()
        print(f"  {feature}: {mean_fn:.2f} (overall: {mean_all:.2f})")

# =============================================================================
# 11. CLINICAL INTERPRETATION
# =============================================================================
print("\n11. CLINICAL INTERPRETATION...")
print("-" * 40)

print("=== CLINICAL INTERPRETATION ===")
print(f"\nModel Performance Summary:")
print(f"• The model correctly identifies {test_recall:.1%} of patients with heart disease (Sensitivity)")
print(f"• The model correctly identifies {specificity:.1%} of patients without heart disease (Specificity)")
print(f"• Of patients predicted to have disease, {test_precision:.1%} actually do (PPV)")
print(f"• Of patients predicted to be healthy, {npv:.1%} actually are (NPV)")
print(f"• Overall accuracy of {test_accuracy:.1%} on unseen patients")

# Interpret AUC
if test_auc > 0.9:
    auc_interpretation = "excellent"
elif test_auc > 0.8:
    auc_interpretation = "good"
elif test_auc > 0.7:
    auc_interpretation = "fair"
else:
    auc_interpretation = "poor"

print(f"• AUC of {test_auc:.3f} indicates {auc_interpretation} discrimination ability")

print(f"\nClinical Impact Assessment:")
print(f"• False Negatives ({len(false_negatives)}): Patients with heart disease missed by the model")
print(f"  - Clinical risk: Delayed treatment for actual heart disease patients")
print(f"  - Rate: {len(false_negatives)/len(y_test[y_test==1]):.1%} of diseased patients missed")

print(f"• False Positives ({len(false_positives)}): Healthy patients incorrectly flagged")
print(f"  - Clinical impact: Unnecessary anxiety, additional testing, healthcare costs")
print(f"  - Rate: {len(false_positives)/len(y_test[y_test==0]):.1%} of healthy patients flagged")

print(f"\nKey Risk Factors Identified (Top 3):")
for i, (_, row) in enumerate(test_feature_importance.head(3).iterrows()):
    if row['Coefficient'] > 0:
        impact = "High risk factor"
        interpretation = f"increases odds by {(row['Odds_Ratio']-1)*100:.1f}%"
    else:
        impact = "Protective factor"
        interpretation = f"decreases odds by {(1-row['Odds_Ratio'])*100:.1f}%"
    
    print(f"  {i+1}. {row['Feature']}: {impact}")
    print(f"     {interpretation} (OR: {row['Odds_Ratio']:.2f})")

print(f"\nRecommended Clinical Use:")
if test_auc >= 0.8 and test_recall >= 0.8:
    print("✓ Model shows good performance for clinical screening")
    print("✓ Can be used as a decision support tool")
elif test_auc >= 0.7:
    print("⚠ Model shows moderate performance")
    print("⚠ Consider as screening tool with expert oversight")
else:
    print("❌ Model performance may be insufficient for clinical use")
    print("❌ Requires improvement before clinical deployment")

print(f"\nOptimal Threshold Recommendations:")
print(f"• For screening (high sensitivity): Use threshold {opt_youden_thresh:.2f}")
print(f"• For diagnosis (balanced): Use threshold {opt_f1_thresh:.2f}")
print(f"• For confirmation (high specificity): Consider higher threshold ~0.7-0.8")

# =============================================================================
# 12. SAVE TEST RESULTS
# =============================================================================
print("\n12. SAVE TEST RESULTS...")
print("-" * 40)

# Save comprehensive test results
test_results = {
    'test_metrics': {
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_score': float(test_f1),
        'auc_roc': float(test_auc),
        'auc_pr': float(pr_auc),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'ppv': float(ppv),
        'npv': float(npv)
    },
    'confusion_matrix': test_cm.tolist(),
    'optimal_thresholds': {
        'best_f1': float(opt_f1_thresh),
        'best_accuracy': float(opt_acc_thresh),
        'best_youden': float(opt_youden_thresh)
    },
    'threshold_analysis': threshold_df.to_dict('records'),
    'feature_importance': test_feature_importance.to_dict('records'),
    'generalization_analysis': {
        'training_cv_auc': float(model_metrics['cv_auc_mean']),
        'training_cv_auc_std': float(model_metrics['cv_auc_std']),
        'test_auc': float(test_auc),
        'auc_difference': float(auc_diff),
        'generalization_status': "good" if abs(auc_diff) < 0.05 else ("overfitting" if auc_diff > 0.05 else "underfitting")
    },
    'misclassification_analysis': {
        'total_misclassified': int(len(misclassified_indices)),
        'false_positives': int(len(false_positives)),
        'false_negatives': int(len(false_negatives)),
        'false_positive_rate': float(len(false_positives)/len(y_test[y_test==0])),
        'false_negative_rate': float(len(false_negatives)/len(y_test[y_test==1]))
    },
    'clinical_interpretation': {
        'auc_interpretation': auc_interpretation,
        'recommended_threshold': float(opt_youden_thresh),
        'clinical_readiness': test_auc >= 0.8 and test_recall >= 0.8
    }
}

joblib.dump(test_results, os.path.join(script_dir, 'test_results.pkl'))
print("✓ Test results saved as 'test_results.pkl'")

# Save threshold analysis as CSV for easy review
threshold_df.to_csv(os.path.join(script_dir, 'threshold_analysis.csv'), index=False)
print("✓ Threshold analysis saved as 'threshold_analysis.csv'")

# Save feature importance as CSV
test_feature_importance.to_csv(os.path.join(script_dir, 'feature_importance.csv'), index=False)
print("✓ Feature importance saved as 'feature_importance.csv'")

# =============================================================================
# 13. TESTING SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("TESTING SUMMARY")
print("=" * 60)

print(f"✓ Model Testing Completed Successfully!")
print(f"✓ Test set size: {len(y_test)} samples")
print(f"✓ Overall performance:")
print(f"  - Accuracy: {test_accuracy:.4f}")
print(f"  - AUC-ROC: {test_auc:.4f} ({auc_interpretation})")
print(f"  - Sensitivity: {test_recall:.4f}")
print(f"  - Specificity: {specificity:.4f}")

print(f"✓ Generalization: {test_results['generalization_analysis']['generalization_status']}")
print(f"  - Training CV AUC: {model_metrics['cv_auc_mean']:.4f}")
print(f"  - Test AUC: {test_auc:.4f}")
print(f"  - Difference: {auc_diff:.4f}")

print(f"✓ Clinical readiness: {'Yes' if test_results['clinical_interpretation']['clinical_readiness'] else 'Needs improvement'}")

print(f"\n✓ Files generated:")
print(f"  - test_confusion_matrix.png (confusion matrix visualization)")
print(f"  - test_roc_pr_curves.png (ROC and PR curves)")
print(f"  - threshold_analysis.png (threshold optimization)")
print(f"  - test_feature_coefficients.png (feature importance)")
print(f"  - test_results.pkl (comprehensive results)")
print(f"  - threshold_analysis.csv (threshold metrics)")
print(f"  - feature_importance.csv (feature rankings)")

print(f"\n✓ Recommended next steps:")
if test_results['clinical_interpretation']['clinical_readiness']:
    print(f"  - Model is ready for clinical validation")
    print(f"  - Consider deployment with threshold = {opt_youden_thresh:.2f}")
    print(f"  - Implement monitoring for model performance")
else:
    print(f"  - Consider model improvements (feature engineering, hyperparameters)")
    print(f"  - Collect more training data if possible")
    print(f"  - Evaluate alternative algorithms")

print(f"\n" + "=" * 60)
print("TESTING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)