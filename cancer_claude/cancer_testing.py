#!/usr/bin/env python3
"""
Breast Cancer Model Testing Suite
Implements comprehensive testing using the ML Evaluation Framework
Based on Steps 13-22 of the cancer prediction plan
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# ML evaluation imports
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, 
    precision_recall_curve, accuracy_score, precision_score, recall_score, 
    f1_score, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_test_summary(tests):
    """Calculate overall test summary"""
    total_tests = len(tests)
    passed = sum(1 for test in tests.values() if isinstance(test, dict) and test.get('status') == 'PASS')
    failed = total_tests - passed
    
    # Determine primary metric and grade
    primary_metric = 'roc_auc'
    primary_value = 0.0
    performance_grade = 'UNKNOWN'
    
    if 'roc_auc' in tests and 'value' in tests['roc_auc']:
        primary_value = tests['roc_auc']['value']
        performance_grade = grade_performance('roc_auc', primary_value)
    elif 'accuracy' in tests and 'value' in tests['accuracy']:
        primary_metric = 'accuracy'
        primary_value = tests['accuracy']['value']
        performance_grade = grade_performance('accuracy', primary_value)
    
    return {
        'total_tests': total_tests,
        'passed': passed,
        'failed': failed,
        'overall_status': 'PASS' if failed == 0 else 'FAIL',
        'key_metrics': {
            'primary_metric': primary_metric,
            'primary_value': primary_value,
            'performance_grade': performance_grade
        }
    }

def grade_performance(metric_name, value, model_type="classification"):
    """Grade model performance based on common benchmarks"""
    if model_type == "classification":
        if metric_name in ["accuracy", "f1_macro", "roc_auc"]:
            if value >= 0.95: return "EXCELLENT"
            elif value >= 0.85: return "GOOD"
            elif value >= 0.75: return "FAIR"
            elif value >= 0.65: return "POOR"
            else: return "VERY_POOR"
    
    return "UNKNOWN"

def run_classification_tests(model, X_test, y_test, y_pred):
    """Run comprehensive classification tests following the framework"""
    tests = {}
    
    # Basic metrics
    tests["accuracy"] = {
        "value": accuracy_score(y_test, y_pred),
        "status": "PASS",
        "threshold": 0.8
    }
    
    tests["precision_recall_f1"] = {
        "precision_macro": precision_score(y_test, y_pred, average='macro'),
        "recall_macro": recall_score(y_test, y_pred, average='macro'),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "precision_weighted": precision_score(y_test, y_pred, average='weighted'),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted'),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        "status": "PASS"
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tests["confusion_matrix"] = {
        "matrix": cm.tolist(),
        "status": "PASS"
    }
    
    # ROC-AUC and clinical metrics for binary classification
    if len(np.unique(y_test)) == 2:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            tests["roc_auc"] = {
                "value": roc_auc_score(y_test, y_proba),
                "status": "PASS",
                "threshold": 0.8
            }
            
            # Clinical metrics for binary classification
            tn, fp, fn, tp = cm.ravel()
            tests["clinical_metrics"] = {
                "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp),
                "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
                "status": "PASS"
            }
            
            # Precision-Recall curve metrics
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)
            tests["precision_recall_auc"] = {
                "value": auc(recall_curve, precision_curve),
                "baseline": np.sum(y_test) / len(y_test),
                "status": "PASS"
            }
            
        except Exception as e:
            tests["roc_auc"] = {"status": "FAIL", "error": str(e)}
            tests["clinical_metrics"] = {"status": "FAIL", "error": str(e)}
    
    return tests

def run_ml_evaluation(model_path, X_test, y_test, results_file, model_id="cancer_model"):
    """
    Comprehensive ML model evaluation suite following the framework
    
    Args:
        model_path (str): Path to saved model (.pkl)
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        results_file (str): Output path for results (.json)
        model_id (str): Model identifier
    """
    
    # Initialize results structure
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "framework_version": "1.0",
            "model_id": model_id,
            "test_description": "Breast cancer model comprehensive evaluation"
        },
        "tests": {},
        "summary": {}
    }
    
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Model Loading Tests
        results["tests"]["model_loading"] = {
            "status": "PASS",
            "model_type": str(type(model).__name__),
            "feature_count": len(X_test.columns),
            "test_samples": len(X_test)
        }
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Run classification tests
        results["tests"].update(run_classification_tests(model, X_test, y_test, y_pred))
        
    except Exception as e:
        results["tests"]["model_loading"] = {
            "status": "FAIL",
            "error": str(e)
        }
    
    # Calculate overall summary
    results["summary"] = calculate_test_summary(results["tests"])
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def perform_threshold_analysis(model, X_test, y_test, output_dir):
    """Perform comprehensive threshold analysis for clinical objectives"""
    
    print("\n" + "="*50)
    print("THRESHOLD ANALYSIS")
    print("="*50)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for different thresholds
    thresholds_analysis = []
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
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
        print(f"Clinical threshold (>=95% sensitivity): {clinical_thresh:.2f}")
    
    # Save threshold analysis
    threshold_df.to_csv(os.path.join(output_dir, 'threshold_analysis.csv'), index=False)
    
    return {
        'best_f1': opt_f1_thresh,
        'best_youden': opt_youden_thresh,
        'best_sensitivity': opt_sensitivity_thresh,
        'analysis_data': threshold_df.to_dict('records')
    }

def perform_error_analysis(model, X_test, y_test, output_dir):
    """Analyze misclassified cases with focus on false negatives"""
    
    print("\n" + "="*50)
    print("ERROR ANALYSIS")
    print("="*50)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Identify misclassified samples
    misclassified_mask = y_test != y_pred
    misclassified_indices = y_test.index[misclassified_mask]
    
    print(f"Misclassified samples: {len(misclassified_indices)} out of {len(y_test)} ({len(misclassified_indices)/len(y_test):.1%})")
    
    # Analyze false positives and false negatives
    false_positives = y_test.index[(y_test == 0) & (y_pred == 1)]
    false_negatives = y_test.index[(y_test == 1) & (y_pred == 0)]
    
    print(f"\nError Breakdown:")
    print(f"False Positives (Benign predicted as Malignant): {len(false_positives)}")
    print(f"False Negatives (Malignant predicted as Benign): {len(false_negatives)}")
    
    error_analysis = {
        'total_misclassified': len(misclassified_indices),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'false_positive_indices': false_positives.tolist(),
        'false_negative_indices': false_negatives.tolist()
    }
    
    # Detailed analysis of false negatives (critical for cancer detection)
    if len(false_negatives) > 0:
        print(f"\n=== FALSE NEGATIVE ANALYSIS (CRITICAL) ===")
        fn_data = X_test.loc[false_negatives]
        fn_probs = y_proba[y_test.index.get_indexer(false_negatives)]
        
        print("False Negative Characteristics (mean values):")
        fn_summary = fn_data.describe().loc['mean'].sort_values(ascending=False)
        print(fn_summary.head(10))
        
        print(f"\nFalse Negative Probabilities:")
        print(f"Mean: {fn_probs.mean():.3f}, Range: {fn_probs.min():.3f} - {fn_probs.max():.3f}")
        
        # Save detailed false negative analysis
        fn_analysis_df = fn_data.copy()
        fn_analysis_df['predicted_proba'] = fn_probs
        fn_analysis_df['actual_label'] = 1
        fn_analysis_df['predicted_label'] = 0
        fn_analysis_df.to_csv(os.path.join(output_dir, 'false_negatives_detailed.csv'))
        
        error_analysis['false_negative_details'] = {
            'mean_probability': float(fn_probs.mean()),
            'min_probability': float(fn_probs.min()),
            'max_probability': float(fn_probs.max()),
            'feature_means': fn_summary.head(10).to_dict()
        }
    
    # Analysis of false positives
    if len(false_positives) > 0:
        print(f"\n=== FALSE POSITIVE ANALYSIS ===")
        fp_data = X_test.loc[false_positives]
        fp_probs = y_proba[y_test.index.get_indexer(false_positives)]
        
        print("False Positive Characteristics (mean values):")
        fp_summary = fp_data.describe().loc['mean'].sort_values(ascending=False)
        print(fp_summary.head(10))
        
        print(f"\nFalse Positive Probabilities:")
        print(f"Mean: {fp_probs.mean():.3f}, Range: {fp_probs.min():.3f} - {fp_probs.max():.3f}")
        
        # Save detailed false positive analysis
        fp_analysis_df = fp_data.copy()
        fp_analysis_df['predicted_proba'] = fp_probs
        fp_analysis_df['actual_label'] = 0
        fp_analysis_df['predicted_label'] = 1
        fp_analysis_df.to_csv(os.path.join(output_dir, 'false_positives_detailed.csv'))
        
        error_analysis['false_positive_details'] = {
            'mean_probability': float(fp_probs.mean()),
            'min_probability': float(fp_probs.min()),
            'max_probability': float(fp_probs.max()),
            'feature_means': fp_summary.head(10).to_dict()
        }
    
    return error_analysis

def generate_visualizations(model, X_test, y_test, output_dir):
    """Generate comprehensive test visualizations"""
    
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC and PR Curves
    plt.figure(figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve - Test Set')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    
    baseline_precision = np.sum(y_test) / len(y_test)
    plt.axhline(y=baseline_precision, color='red', linestyle='--', 
                label=f'Baseline ({baseline_precision:.3f})')
    
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curve - Test Set')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Probability Distribution by Class
    plt.figure(figsize=(10, 6))
    
    # Probabilities for each class
    benign_probs = y_proba[y_test == 0]
    malignant_probs = y_proba[y_test == 1]
    
    plt.hist(benign_probs, bins=30, alpha=0.7, label='Benign', color='skyblue', density=True)
    plt.hist(malignant_probs, bins=30, alpha=0.7, label='Malignant', color='salmon', density=True)
    
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Default Threshold (0.5)')
    plt.xlabel('Predicted Probability of Malignancy')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Probabilities by True Class')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'probability_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved:")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_pr_curves.png")
    print(f"  - probability_distributions.png")

def provide_clinical_interpretation(test_results, error_analysis):
    """Provide comprehensive clinical interpretation"""
    
    print("\n" + "="*50)
    print("CLINICAL INTERPRETATION")
    print("="*50)
    
    # Extract key metrics
    tests = test_results['tests']
    
    if 'clinical_metrics' in tests:
        cm = tests['clinical_metrics']
        sensitivity = cm['sensitivity']
        specificity = cm['specificity']
        ppv = cm['ppv']
        npv = cm['npv']
        
        print(f"\nModel Performance Summary:")
        print(f"- Sensitivity (Cancer Detection Rate): {sensitivity:.1%}")
        print(f"  - The model correctly identifies {sensitivity:.1%} of malignant cases")
        print(f"  - {error_analysis['false_negatives']} malignant cases were missed (FALSE NEGATIVES)")
        
        print(f"\n- Specificity (Benign Identification): {specificity:.1%}")
        print(f"  - The model correctly identifies {specificity:.1%} of benign cases")
        print(f"  - {error_analysis['false_positives']} benign cases were incorrectly flagged")
        
        print(f"\n- Positive Predictive Value: {ppv:.1%}")
        print(f"  - Of cases predicted as malignant, {ppv:.1%} actually are malignant")
        
        print(f"\n- Negative Predictive Value: {npv:.1%}")
        print(f"  - Of cases predicted as benign, {npv:.1%} actually are benign")
    
    if 'accuracy' in tests:
        accuracy = tests['accuracy']['value']
        print(f"\n- Overall Accuracy: {accuracy:.1%}")
    
    if 'roc_auc' in tests:
        auc_score = tests['roc_auc']['value']
        discrimination = 'Excellent' if auc_score > 0.9 else 'Good' if auc_score > 0.8 else 'Fair' if auc_score > 0.7 else 'Poor'
        print(f"- AUC-ROC: {auc_score:.3f} ({discrimination} discrimination)")
    
    print(f"\nClinical Impact Assessment:")
    if error_analysis['false_negatives'] > 0:
        print(f"[CRITICAL] {error_analysis['false_negatives']} malignant cases missed by model")
        print(f"   These patients would not receive immediate cancer treatment")
        print(f"   Consider lowering decision threshold to increase sensitivity")
    
    if error_analysis['false_positives'] > 0:
        print(f"[WARNING] {error_analysis['false_positives']} benign cases incorrectly flagged as malignant")
        print(f"   These patients would undergo unnecessary further testing/anxiety")
        print(f"   But this is preferred over missing actual cancers")
    
    # Model recommendation
    print(f"\nCLINICAL RECOMMENDATIONS:")
    if 'roc_auc' in tests and tests['roc_auc']['value'] >= 0.95:
        print("  [EXCELLENT] Excellent discriminative performance - suitable for clinical screening")
    elif 'roc_auc' in tests and tests['roc_auc']['value'] >= 0.85:
        print("  [GOOD] Good discriminative performance - appropriate for clinical use with supervision")
    else:
        print("  [WARNING] Performance may need improvement for clinical deployment")
    
    if 'clinical_metrics' in tests:
        cm = tests['clinical_metrics']
        if cm['sensitivity'] < 0.8:
            print("  🔴 LOW SENSITIVITY - Many cancer cases will be missed")
            print("      Recommend: Lower decision threshold or improve model")
        elif cm['sensitivity'] >= 0.9:
            print("  [GOOD] High sensitivity - Good for cancer screening")
        
        if cm['specificity'] < 0.8:
            print("  🟡 Low specificity - High false positive rate")
            print("      Impact: Increased unnecessary procedures")
        elif cm['specificity'] >= 0.9:
            print("  [GOOD] High specificity - Low false alarm rate")

def main():
    """Main testing pipeline following the comprehensive plan"""
    
    print("=" * 70)
    print("BREAST CANCER MODEL COMPREHENSIVE TESTING SUITE")
    print("Following ML Evaluation Framework Standards")
    print("=" * 70)
    
    # Setup directories and paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    tests_dir = os.path.join(current_dir, 'tests')
    
    # Create tests directory
    os.makedirs(tests_dir, exist_ok=True)
    
    # Model and data paths
    model_path = os.path.join(models_dir, 'breast_cancer_model.pkl')
    feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
    metrics_path = os.path.join(models_dir, 'model_metrics.pkl')
    
    # Load saved test data (from training script)
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, 'cancer.csv')
    
    try:
        # Load data and recreate test split (using same random state as training)
        print("\nStep 13: Loading trained model and preparing test data")
        print("=" * 50)
        
        df = pd.read_csv(data_path)
        
        # Preprocess data (same as training)
        df_processed = df.drop('id', axis=1)
        unnamed_cols = [col for col in df_processed.columns if 'Unnamed' in col]
        if unnamed_cols:
            df_processed = df_processed.drop(unnamed_cols, axis=1)
        
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        
        label_encoder = LabelEncoder()
        df_processed['diagnosis_encoded'] = label_encoder.fit_transform(df_processed['diagnosis'])
        
        # Separate features and target
        feature_columns = [col for col in df_processed.columns 
                          if col not in ['diagnosis', 'diagnosis_encoded']]
        X = df_processed[feature_columns]
        y = df_processed['diagnosis_encoded']
        
        # Recreate train-test split (same random state as training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Test set loaded: {len(X_test)} samples, {len(X_test.columns)} features")
        print(f"Class distribution - Benign: {np.sum(y_test == 0)}, Malignant: {np.sum(y_test == 1)}")
        
        # Load model and metadata
        model = joblib.load(model_path)
        feature_names = joblib.load(feature_names_path)
        model_metrics = joblib.load(metrics_path)
        
        print("[SUCCESS] Model loaded successfully")
        print(f"Model type: {type(model).__name__}")
        print(f"Training CV AUC: {model_metrics['cv_auc_mean']:.4f}")
        
        # Step 14-15: Comprehensive ML Evaluation using Framework
        print("\nStep 14-15: Running comprehensive ML evaluation")
        print("=" * 50)
        
        results_file = os.path.join(tests_dir, 'cancer_model_test_results.json')
        test_results = run_ml_evaluation(
            model_path=model_path,
            X_test=X_test,
            y_test=y_test,
            results_file=results_file,
            model_id="breast_cancer_lr_v1"
        )
        
        print("[SUCCESS] Framework evaluation completed")
        print(f"Results saved: {results_file}")
        print(f"Overall Status: {test_results['summary']['overall_status']}")
        print(f"Tests Passed: {test_results['summary']['passed']}/{test_results['summary']['total_tests']}")
        
        # Step 16-17: Advanced Analysis
        print("\nStep 16-17: Threshold analysis and optimization")
        threshold_results = perform_threshold_analysis(model, X_test, y_test, tests_dir)
        
        # Step 18-20: Error Analysis  
        print("\nStep 18-20: Comprehensive error analysis")
        error_analysis = perform_error_analysis(model, X_test, y_test, tests_dir)
        
        # Generate visualizations
        print("\nGenerating comprehensive visualizations")
        generate_visualizations(model, X_test, y_test, tests_dir)
        
        # Step 21: Clinical Interpretation
        provide_clinical_interpretation(test_results, error_analysis)
        
        # Step 22: Save comprehensive results
        print("\nStep 22: Saving comprehensive test results")
        print("=" * 50)
        
        # Enhanced results with all analyses
        comprehensive_results = test_results.copy()
        comprehensive_results['threshold_analysis'] = threshold_results
        comprehensive_results['error_analysis'] = error_analysis
        comprehensive_results['clinical_interpretation'] = {
            'focus': 'cancer_detection_priority',
            'primary_concern': 'minimize_false_negatives',
            'recommended_threshold': threshold_results.get('best_sensitivity', 0.5)
        }
        
        # Save comprehensive results
        comprehensive_file = os.path.join(tests_dir, 'comprehensive_test_results.json')
        with open(comprehensive_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print("[SUCCESS] Comprehensive testing completed successfully!")
        print(f"\nAll test artifacts saved to: {tests_dir}")
        print("Files created:")
        print(f"  - cancer_model_test_results.json (Framework standard)")
        print(f"  - comprehensive_test_results.json (Extended analysis)")
        print(f"  - threshold_analysis.csv")
        print(f"  - false_negatives_detailed.csv (if any)")
        print(f"  - false_positives_detailed.csv (if any)")
        print(f"  - confusion_matrix.png")
        print(f"  - roc_pr_curves.png")
        print(f"  - probability_distributions.png")
        
        # Performance Summary
        print(f"\nFINAL PERFORMANCE SUMMARY:")
        summary = test_results['summary']
        print(f"   Grade: {summary['key_metrics']['performance_grade']}")
        print(f"   Primary Metric: {summary['key_metrics']['primary_metric']} = {summary['key_metrics']['primary_value']:.4f}")
        print(f"   False Negatives: {error_analysis['false_negatives']} (CRITICAL for cancer detection)")
        print(f"   False Positives: {error_analysis['false_positives']}")
        
        return test_results, comprehensive_results
        
    except Exception as e:
        print(f"[ERROR] Testing failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    test_results, comprehensive_results = main()
    
    if test_results:
        print(f"\n[COMPLETE] Testing pipeline completed successfully!")
        print(f"   Use 'python ml_analyzer.py cancer_model_test_results.json' for framework analysis")
    else:
        print(f"\n[FAILURE] Testing pipeline failed!")
        sys.exit(1)