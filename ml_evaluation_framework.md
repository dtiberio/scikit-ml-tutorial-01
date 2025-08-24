<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright © 2025 github.com/dtiberio
-->

# ML Model Evaluation Framework

A standardized framework for evaluating and comparing supervised machine learning models (Linear Regression, Logistic Regression, etc.) with consistent testing protocols and result formats.

## Section 1: Test Suite and Results Format

### 1.1 Core Test Categories

#### Model Loading and Validation Tests
- **Model Integrity**: Verify model loads correctly from disk
- **Feature Compatibility**: Ensure test data matches training features
- **Prediction Functionality**: Basic prediction capability test
- **Model Metadata**: Extract and validate model configuration

#### Performance Metrics Tests

**For Classification Models:**
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class and macro/weighted averages
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification breakdown
- **Sensitivity/Specificity**: Clinical metrics for binary classification

**For Regression Models:**
- **MSE/RMSE**: Mean squared error metrics
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination
- **Residual Analysis**: Distribution and patterns

#### Robustness Tests
- **Cross-Validation**: K-fold performance consistency
- **Threshold Analysis**: Optimal decision boundaries (classification)
- **Feature Importance**: Model interpretability
- **Prediction Confidence**: Probability distributions

### 1.2 Test Implementation Example

```python
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import *
import json
from datetime import datetime

def run_ml_evaluation(model_path, test_data_path, results_file):
    """
    Comprehensive ML model evaluation suite
    
    Args:
        model_path (str): Path to saved model (.pkl)
        test_data_path (str): Path to test dataset (.csv)
        results_file (str): Output path for results (.json)
    """
    
    # Initialize results structure
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "test_data_path": test_data_path,
            "framework_version": "1.0"
        },
        "tests": {},
        "summary": {}
    }
    
    try:
        # Load model and data
        model = joblib.load(model_path)
        test_data = pd.read_csv(test_data_path)
        
        # Extract features and target (assumes last column is target)
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        
        # Model Loading Tests
        results["tests"]["model_loading"] = {
            "status": "PASS",
            "model_type": str(type(model).__name__),
            "feature_count": len(X_test.columns),
            "test_samples": len(X_test)
        }
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Determine model type and run appropriate tests
        if hasattr(model, 'predict_proba'):
            # Classification model
            results["tests"].update(run_classification_tests(model, X_test, y_test, y_pred))
        else:
            # Regression model
            results["tests"].update(run_regression_tests(model, X_test, y_test, y_pred))
            
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

def run_classification_tests(model, X_test, y_test, y_pred):
    """Run classification-specific tests"""
    tests = {}
    
    # Basic metrics
    tests["accuracy"] = {
        "value": accuracy_score(y_test, y_pred),
        "status": "PASS"
    }
    
    tests["precision_recall_f1"] = {
        "precision_macro": precision_score(y_test, y_pred, average='macro'),
        "recall_macro": recall_score(y_test, y_pred, average='macro'),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "status": "PASS"
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tests["confusion_matrix"] = {
        "matrix": cm.tolist(),
        "status": "PASS"
    }
    
    # ROC-AUC if binary classification
    if len(np.unique(y_test)) == 2:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            tests["roc_auc"] = {
                "value": roc_auc_score(y_test, y_proba),
                "status": "PASS"
            }
            
            # Clinical metrics for binary classification
            tn, fp, fn, tp = cm.ravel()
            tests["clinical_metrics"] = {
                "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp),
                "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
                "status": "PASS"
            }
        except Exception as e:
            tests["roc_auc"] = {"status": "FAIL", "error": str(e)}
    
    return tests

def run_regression_tests(model, X_test, y_test, y_pred):
    """Run regression-specific tests"""
    tests = {}
    
    tests["mse_rmse"] = {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "status": "PASS"
    }
    
    tests["mae"] = {
        "value": mean_absolute_error(y_test, y_pred),
        "status": "PASS"
    }
    
    tests["r2_score"] = {
        "value": r2_score(y_test, y_pred),
        "status": "PASS"
    }
    
    return tests
```

### 1.3 Test Results File Format

The framework outputs results in JSON format for both human readability and script parsing:

```json
{
  "metadata": {
    "timestamp": "2024-01-15T14:30:00.123456",
    "model_path": "models/cancer_model.pkl",
    "test_data_path": "data/test_set.csv",
    "framework_version": "1.0",
    "model_id": "cancer_lr_v1",
    "test_description": "Breast cancer model evaluation"
  },
  "tests": {
    "model_loading": {
      "status": "PASS",
      "model_type": "Pipeline",
      "feature_count": 30,
      "test_samples": 114
    },
    "accuracy": {
      "value": 0.9649,
      "status": "PASS",
      "threshold": 0.8
    },
    "precision_recall_f1": {
      "precision_macro": 0.9621,
      "recall_macro": 0.9583,
      "f1_macro": 0.9599,
      "status": "PASS"
    },
    "roc_auc": {
      "value": 0.9876,
      "status": "PASS",
      "threshold": 0.8
    },
    "clinical_metrics": {
      "sensitivity": 0.9512,
      "specificity": 0.9722,
      "ppv": 0.9512,
      "npv": 0.9722,
      "status": "PASS"
    },
    "confusion_matrix": {
      "matrix": [[70, 2], [2, 40]],
      "status": "PASS"
    }
  },
  "summary": {
    "total_tests": 6,
    "passed": 6,
    "failed": 0,
    "overall_status": "PASS",
    "key_metrics": {
      "primary_metric": "roc_auc",
      "primary_value": 0.9876,
      "performance_grade": "EXCELLENT"
    }
  }
}
```

### 1.4 Performance Grading System

```python
def grade_performance(metric_name, value, model_type="classification"):
    """
    Grade model performance based on common benchmarks
    """
    if model_type == "classification":
        if metric_name in ["accuracy", "f1_macro", "roc_auc"]:
            if value >= 0.95: return "EXCELLENT"
            elif value >= 0.85: return "GOOD"
            elif value >= 0.75: return "FAIR"
            elif value >= 0.65: return "POOR"
            else: return "VERY_POOR"
    
    elif model_type == "regression":
        if metric_name == "r2_score":
            if value >= 0.9: return "EXCELLENT"
            elif value >= 0.8: return "GOOD"
            elif value >= 0.7: return "FAIR"
            elif value >= 0.6: return "POOR"
            else: return "VERY_POOR"
    
    return "UNKNOWN"
```

---

## Section 2: Test Results Analysis and Comparison Script

### 2.1 Results Interpreter

The framework includes a comprehensive script to analyze and compare test results:

```python
#!/usr/bin/env python3
"""
ML Evaluation Results Analyzer
Compares multiple test results and provides recommendations
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class MLResultsAnalyzer:
    def __init__(self):
        self.results = []
        self.comparison_metrics = [
            "accuracy", "roc_auc", "precision_macro", 
            "recall_macro", "f1_macro", "sensitivity", "specificity"
        ]
    
    def load_results(self, result_files: List[str]):
        """Load multiple test result files"""
        self.results = []
        for file_path in result_files:
            with open(file_path, 'r') as f:
                result = json.load(f)
                result['file_path'] = file_path
                self.results.append(result)
        
        print(f"Loaded {len(self.results)} test result files")
    
    def analyze_single_result(self, result_file: str):
        """Analyze a single test result file"""
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        print("\n" + "="*50)
        print("SINGLE MODEL ANALYSIS")
        print("="*50)
        
        # Basic info
        metadata = result['metadata']
        print(f"Model: {metadata.get('model_id', 'Unknown')}")
        print(f"Tested: {metadata['timestamp']}")
        print(f"Test samples: {result['tests']['model_loading']['test_samples']}")
        
        # Performance summary
        summary = result['summary']
        print(f"\nOverall Status: {summary['overall_status']}")
        print(f"Tests Passed: {summary['passed']}/{summary['total_tests']}")
        
        # Key metrics
        key_metrics = summary.get('key_metrics', {})
        if key_metrics:
            print(f"Primary Metric ({key_metrics['primary_metric']}): {key_metrics['primary_value']:.4f}")
            print(f"Performance Grade: {key_metrics['performance_grade']}")
        
        # Detailed metrics
        print("\nDetailed Metrics:")
        tests = result['tests']
        
        if 'accuracy' in tests:
            print(f"  Accuracy: {tests['accuracy']['value']:.4f}")
        
        if 'roc_auc' in tests:
            print(f"  ROC-AUC: {tests['roc_auc']['value']:.4f}")
        
        if 'clinical_metrics' in tests:
            cm = tests['clinical_metrics']
            print(f"  Sensitivity: {cm['sensitivity']:.4f}")
            print(f"  Specificity: {cm['specificity']:.4f}")
        
        # Recommendations
        self._provide_single_model_recommendations(result)
        
        return result
    
    def compare_results(self):
        """Compare multiple test results"""
        if len(self.results) < 2:
            print("Need at least 2 results for comparison")
            return
        
        print("\n" + "="*60)
        print("MULTI-MODEL COMPARISON")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        
        for i, result in enumerate(self.results):
            model_data = {
                'Model': result['metadata'].get('model_id', f'Model_{i+1}'),
                'File': Path(result['file_path']).name,
                'Status': result['summary']['overall_status'],
                'Tests_Passed': f"{result['summary']['passed']}/{result['summary']['total_tests']}"
            }
            
            # Extract key metrics
            tests = result['tests']
            for metric in self.comparison_metrics:
                if metric in tests:
                    if isinstance(tests[metric], dict):
                        if 'value' in tests[metric]:
                            model_data[metric] = tests[metric]['value']
                        else:
                            # Handle nested metrics
                            for key, value in tests[metric].items():
                                if key != 'status' and isinstance(value, (int, float)):
                                    model_data[f"{metric}_{key}"] = value
            
            comparison_data.append(model_data)
        
        # Display comparison table
        df = pd.DataFrame(comparison_data)
        print("\nModel Comparison Summary:")
        print(df.to_string(index=False, float_format='{:.4f}'.format))
        
        # Rank models
        ranking = self._rank_models(df)
        print(f"\n🏆 RECOMMENDED MODEL: {ranking[0]['model']}")
        print(f"   Primary Metric: {ranking[0]['primary_metric']:.4f}")
        print(f"   Reason: {ranking[0]['reason']}")
        
        # Show runner-ups
        if len(ranking) > 1:
            print(f"\n🥈 Runner-up: {ranking[1]['model']}")
            print(f"   Difference: {ranking[0]['primary_metric'] - ranking[1]['primary_metric']:.4f}")
        
        return df, ranking
    
    def _rank_models(self, df: pd.DataFrame) -> List[Dict]:
        """Rank models based on performance"""
        rankings = []
        
        for _, row in df.iterrows():
            score = 0
            primary_metric_val = 0
            
            # Weight different metrics
            if 'roc_auc' in row and pd.notna(row['roc_auc']):
                score += row['roc_auc'] * 0.4
                primary_metric_val = row['roc_auc']
            
            if 'f1_macro' in row and pd.notna(row['f1_macro']):
                score += row['f1_macro'] * 0.3
            
            if 'accuracy' in row and pd.notna(row['accuracy']):
                score += row['accuracy'] * 0.3
            
            rankings.append({
                'model': row['Model'],
                'score': score,
                'primary_metric': primary_metric_val,
                'reason': self._get_ranking_reason(row)
            })
        
        # Sort by score descending
        rankings.sort(key=lambda x: x['score'], reverse=True)
        return rankings
    
    def _get_ranking_reason(self, row) -> str:
        """Generate reasoning for model ranking"""
        reasons = []
        
        if 'roc_auc' in row and row['roc_auc'] >= 0.95:
            reasons.append("excellent discriminative ability")
        
        if 'sensitivity' in row and row['sensitivity'] >= 0.9:
            reasons.append("high sensitivity (good for screening)")
        
        if 'specificity' in row and row['specificity'] >= 0.9:
            reasons.append("high specificity (low false positives)")
        
        if row.get('Status') == 'PASS':
            reasons.append("all tests passed")
        
        return ", ".join(reasons) if reasons else "baseline performance"
    
    def _provide_single_model_recommendations(self, result):
        """Provide recommendations for a single model"""
        print("\n📋 RECOMMENDATIONS:")
        
        tests = result['tests']
        
        # Check for potential issues
        if 'roc_auc' in tests and tests['roc_auc']['value'] < 0.8:
            print("  ⚠️  Consider feature engineering or different algorithm")
        
        if 'clinical_metrics' in tests:
            cm = tests['clinical_metrics']
            if cm['sensitivity'] < 0.8:
                print("  ⚠️  Low sensitivity - many positive cases missed")
            if cm['specificity'] < 0.8:
                print("  ⚠️  Low specificity - high false positive rate")
        
        # Positive feedback
        if result['summary']['overall_status'] == 'PASS':
            print("  ✅ Model meets basic performance criteria")
        
        if 'roc_auc' in tests and tests['roc_auc']['value'] >= 0.95:
            print("  🎯 Excellent discriminative performance")

# Usage example
def main():
    analyzer = MLResultsAnalyzer()
    
    # For single model analysis
    if len(sys.argv) == 2:
        analyzer.analyze_single_result(sys.argv[1])
    
    # For comparison
    elif len(sys.argv) > 2:
        analyzer.load_results(sys.argv[1:])
        analyzer.compare_results()
    
    else:
        print("Usage: python ml_analyzer.py <result_file1> [result_file2] [result_file3]")

if __name__ == "__main__":
    import sys
    main()
```

### 2.2 Quick Usage Examples

**Single Model Analysis:**
```bash
python ml_analyzer.py cancer_model_results.json
```

**Compare 3 Models:**
```bash
python ml_analyzer.py model_v1_results.json model_v2_results.json model_v3_results.json
```

**Generate Test Results:**
```python
# Example integration
run_ml_evaluation(
    model_path="models/cancer_model.pkl",
    test_data_path="data/test_set.csv", 
    results_file="results/cancer_v1_results.json"
)
```

### 2.3 Key Features

- **Standardized Format**: Consistent JSON structure for all results
- **Human & Machine Readable**: Easy to read manually or parse programmatically
- **Comprehensive Metrics**: Covers classification and regression scenarios
- **Performance Grading**: Automatic performance classification
- **Intelligent Comparison**: Weighted ranking system for model selection
- **Clinical Focus**: Includes sensitivity/specificity for medical applications
- **Extensible**: Easy to add new tests and metrics

This framework provides a solid foundation for ML model evaluation while remaining simple and practical for everyday use.