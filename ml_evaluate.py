# SPDX-License-Identifier: MIT
# Copyright © 2025 github.com/dtiberio

#!/usr/bin/env python3
"""
ML Evaluation Results Analyzer
Compares multiple test results and provides recommendations based on ML best practices

Usage:
    python ml_evaluate.py <result_file1> [result_file2] [result_file3]

Examples:
    python ml_evaluate.py cancer_model_results.json
    python ml_evaluate.py model_v1.json model_v2.json model_v3.json
"""

import json
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

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
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    result['file_path'] = file_path
                    self.results.append(result)
                print(f"✓ Loaded: {Path(file_path).name}")
            except Exception as e:
                print(f"✗ Failed to load {file_path}: {e}")
        
        print(f"\nSuccessfully loaded {len(self.results)} test result files\n")
    
    def analyze_single_result(self, result_file: str):
        """Analyze a single test result file with ML best practices interpretation"""
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            return None
        
        print("=" * 70)
        print("SINGLE MODEL ANALYSIS - ML EVALUATION REPORT")
        print("=" * 70)
        
        # Basic info
        metadata = result.get('metadata', {})
        print(f"Model ID: {metadata.get('model_id', 'Unknown')}")
        print(f"Tested: {metadata.get('timestamp', 'Unknown')}")
        
        if 'model_loading' in result.get('tests', {}):
            model_info = result['tests']['model_loading']
            print(f"Model Type: {model_info.get('model_type', 'Unknown')}")
            print(f"Features: {model_info.get('feature_count', 'Unknown')}")
            print(f"Test Samples: {model_info.get('test_samples', 'Unknown')}")
        
        # Overall status
        summary = result.get('summary', {})
        print(f"\nOverall Status: {summary.get('overall_status', 'Unknown')}")
        print(f"Tests Passed: {summary.get('passed', 0)}/{summary.get('total_tests', 0)}")
        
        # Performance grade
        key_metrics = summary.get('key_metrics', {})
        if key_metrics:
            grade = key_metrics.get('performance_grade', 'Unknown')
            primary_metric = key_metrics.get('primary_metric', 'Unknown')
            primary_value = key_metrics.get('primary_value', 0)
            print(f"\nPerformance Grade: {grade}")
            print(f"Primary Metric ({primary_metric}): {primary_value:.4f}")
        
        print("\n" + "-" * 50)
        print("DETAILED PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        tests = result.get('tests', {})
        
        # Classification metrics analysis
        self._analyze_classification_metrics(tests)
        
        # Clinical metrics analysis (if available)
        self._analyze_clinical_metrics(tests)
        
        # Error analysis
        self._analyze_confusion_matrix(tests)
        
        print("\n" + "-" * 50)
        print("ML BEST PRACTICES INTERPRETATION")
        print("-" * 50)
        
        self._provide_ml_interpretation(tests, summary)
        
        print("\n" + "-" * 50)
        print("RECOMMENDATIONS")
        print("-" * 50)
        
        self._provide_single_model_recommendations(result)
        
        return result
    
    def compare_results(self):
        """Compare multiple test results with ML best practices"""
        if len(self.results) < 2:
            print("Need at least 2 results for comparison")
            return
        
        print("=" * 80)
        print("MULTI-MODEL COMPARISON ANALYSIS")
        print("=" * 80)
        
        # Create comparison table
        comparison_data = []
        
        for i, result in enumerate(self.results):
            model_data = {
                'Model': result.get('metadata', {}).get('model_id', f'Model_{i+1}'),
                'File': Path(result['file_path']).name,
                'Status': result.get('summary', {}).get('overall_status', 'Unknown'),
                'Grade': result.get('summary', {}).get('key_metrics', {}).get('performance_grade', 'Unknown')
            }
            
            # Extract key metrics
            tests = result.get('tests', {})
            
            if 'accuracy' in tests:
                model_data['Accuracy'] = tests['accuracy'].get('value', 0)
            
            if 'roc_auc' in tests:
                model_data['ROC-AUC'] = tests['roc_auc'].get('value', 0)
            
            if 'precision_recall_f1' in tests:
                prf = tests['precision_recall_f1']
                model_data['F1-Score'] = prf.get('f1_macro', 0)
                model_data['Precision'] = prf.get('precision_macro', 0)
                model_data['Recall'] = prf.get('recall_macro', 0)
            
            if 'clinical_metrics' in tests:
                cm = tests['clinical_metrics']
                model_data['Sensitivity'] = cm.get('sensitivity', 0)
                model_data['Specificity'] = cm.get('specificity', 0)
            
            comparison_data.append(model_data)
        
        # Display comparison table
        df = pd.DataFrame(comparison_data)
        print("\nModel Performance Comparison:")
        print("-" * 80)
        
        # Format the display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(df.to_string(index=False))
        
        # Rank models
        ranking = self._rank_models(df)
        
        print(f"\nBEST MODEL: {ranking[0]['model']}")
        print(f"   Score: {ranking[0]['score']:.4f}")
        print(f"   Strengths: {ranking[0]['reason']}")
        
        # Show detailed comparison
        self._detailed_model_comparison(ranking)
        
        # Provide comparison recommendations
        self._provide_comparison_recommendations(df, ranking)
        
        return df, ranking
    
    def _analyze_classification_metrics(self, tests):
        """Analyze classification metrics with ML best practices"""
        print("\nCLASSIFICATION METRICS:")
        
        if 'accuracy' in tests:
            acc = tests['accuracy'].get('value', 0)
            print(f"   Accuracy: {acc:.4f} ({self._interpret_accuracy(acc)})")
        
        if 'roc_auc' in tests:
            auc = tests['roc_auc'].get('value', 0)
            print(f"   ROC-AUC: {auc:.4f} ({self._interpret_auc(auc)})")
        
        if 'precision_recall_f1' in tests:
            prf = tests['precision_recall_f1']
            f1 = prf.get('f1_macro', 0)
            precision = prf.get('precision_macro', 0)
            recall = prf.get('recall_macro', 0)
            print(f"   F1-Score: {f1:.4f} ({self._interpret_f1(f1)})")
            print(f"   Precision: {precision:.4f} ({self._interpret_precision(precision)})")
            print(f"   Recall: {recall:.4f} ({self._interpret_recall(recall)})")
    
    def _analyze_clinical_metrics(self, tests):
        """Analyze clinical metrics for medical applications"""
        if 'clinical_metrics' not in tests:
            return
        
        print("\nCLINICAL METRICS:")
        cm = tests['clinical_metrics']
        
        sensitivity = cm.get('sensitivity', 0)
        specificity = cm.get('specificity', 0)
        ppv = cm.get('ppv', 0)
        npv = cm.get('npv', 0)
        
        print(f"   Sensitivity: {sensitivity:.4f} ({self._interpret_sensitivity(sensitivity)})")
        print(f"   Specificity: {specificity:.4f} ({self._interpret_specificity(specificity)})")
        print(f"   PPV: {ppv:.4f} ({self._interpret_ppv(ppv)})")
        print(f"   NPV: {npv:.4f} ({self._interpret_npv(npv)})")
    
    def _analyze_confusion_matrix(self, tests):
        """Analyze confusion matrix for error patterns"""
        if 'confusion_matrix' not in tests:
            return
        
        matrix = tests['confusion_matrix'].get('matrix', [])
        if len(matrix) == 2 and len(matrix[0]) == 2:
            tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
            total = tn + fp + fn + tp
            
            print(f"\nERROR ANALYSIS:")
            print(f"   True Negatives: {tn} ({tn/total:.1%})")
            print(f"   False Positives: {fp} ({fp/total:.1%}) - {self._interpret_fp(fp, total)}")
            print(f"   False Negatives: {fn} ({fn/total:.1%}) - {self._interpret_fn(fn, total)}")
            print(f"   True Positives: {tp} ({tp/total:.1%})")
    
    def _provide_ml_interpretation(self, tests, summary):
        """Provide ML best practices interpretation"""
        interpretations = []
        
        # Overall performance assessment
        grade = summary.get('key_metrics', {}).get('performance_grade', '')
        if grade == 'EXCELLENT':
            interpretations.append("Model shows excellent performance suitable for production use")
        elif grade == 'GOOD':
            interpretations.append("Model shows good performance, consider minor optimizations")
        elif grade == 'FAIR':
            interpretations.append("Model shows fair performance, significant improvements needed")
        elif grade == 'POOR':
            interpretations.append("Model shows poor performance, major rework required")
        
        # Specific metric interpretations
        if 'roc_auc' in tests:
            auc = tests['roc_auc'].get('value', 0)
            if auc >= 0.9:
                interpretations.append("Excellent discriminative ability - model can reliably distinguish between classes")
            elif auc >= 0.8:
                interpretations.append("Good discriminative ability with room for improvement")
            elif auc < 0.7:
                interpretations.append("Poor discriminative ability - consider feature engineering or different algorithms")
        
        # Balance assessment
        if 'clinical_metrics' in tests:
            cm = tests['clinical_metrics']
            sens = cm.get('sensitivity', 0)
            spec = cm.get('specificity', 0)
            
            if abs(sens - spec) < 0.1:
                interpretations.append("Well-balanced model with similar sensitivity and specificity")
            elif sens > spec + 0.1:
                interpretations.append("Model favors sensitivity (good for screening, may have more false positives)")
            elif spec > sens + 0.1:
                interpretations.append("Model favors specificity (conservative, may miss some positive cases)")
        
        for interpretation in interpretations:
            print(f"   - {interpretation}")
    
    def _provide_single_model_recommendations(self, result):
        """Provide specific recommendations based on ML best practices"""
        tests = result.get('tests', {})
        recommendations = []
        
        # Performance-based recommendations
        if 'roc_auc' in tests:
            auc = tests['roc_auc'].get('value', 0)
            if auc < 0.7:
                recommendations.append("CRITICAL: Consider feature engineering, different algorithms, or data quality issues")
            elif auc < 0.8:
                recommendations.append("Consider hyperparameter tuning and feature selection")
            elif auc >= 0.95:
                recommendations.append("Excellent performance - ready for production with proper validation")
        
        # Balance recommendations
        if 'clinical_metrics' in tests:
            cm = tests['clinical_metrics']
            sens = cm.get('sensitivity', 0)
            spec = cm.get('specificity', 0)
            
            if sens < 0.8:
                recommendations.append("Low sensitivity - consider lowering decision threshold or class rebalancing")
            if spec < 0.8:
                recommendations.append("Low specificity - consider raising decision threshold or better feature engineering")
            if sens >= 0.9 and spec >= 0.9:
                recommendations.append("Excellent clinical performance for both sensitivity and specificity")
        
        # Overfitting check
        if 'confusion_matrix' in tests:
            matrix = tests['confusion_matrix'].get('matrix', [])
            if len(matrix) == 2:
                total_errors = matrix[0][1] + matrix[1][0]
                total_samples = sum(sum(row) for row in matrix)
                error_rate = total_errors / total_samples if total_samples > 0 else 0
                
                if error_rate < 0.01:
                    recommendations.append("Very low error rate - verify no data leakage or overfitting")
        
        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("   No specific recommendations - model performance is adequate")
    
    def _rank_models(self, df: pd.DataFrame) -> List[Dict]:
        """Rank models based on weighted performance metrics"""
        rankings = []
        
        for _, row in df.iterrows():
            score = 0
            weights_used = 0
            
            # Weighted scoring system based on ML best practices
            if 'ROC-AUC' in row and pd.notna(row['ROC-AUC']):
                score += row['ROC-AUC'] * 0.3  # Primary discriminative metric
                weights_used += 0.3
            
            if 'F1-Score' in row and pd.notna(row['F1-Score']):
                score += row['F1-Score'] * 0.25  # Balanced metric
                weights_used += 0.25
            
            if 'Accuracy' in row and pd.notna(row['Accuracy']):
                score += row['Accuracy'] * 0.2  # Overall correctness
                weights_used += 0.2
            
            if 'Sensitivity' in row and pd.notna(row['Sensitivity']):
                score += row['Sensitivity'] * 0.125  # Clinical importance
                weights_used += 0.125
            
            if 'Specificity' in row and pd.notna(row['Specificity']):
                score += row['Specificity'] * 0.125  # Clinical importance
                weights_used += 0.125
            
            # Normalize score by weights used
            if weights_used > 0:
                score = score / weights_used
            
            rankings.append({
                'model': row['Model'],
                'score': score,
                'primary_metric': row.get('ROC-AUC', row.get('F1-Score', row.get('Accuracy', 0))),
                'reason': self._get_ranking_reason(row)
            })
        
        rankings.sort(key=lambda x: x['score'], reverse=True)
        return rankings
    
    def _get_ranking_reason(self, row) -> str:
        """Generate reasoning for model ranking based on ML criteria"""
        reasons = []
        
        if 'ROC-AUC' in row and pd.notna(row['ROC-AUC']):
            if row['ROC-AUC'] >= 0.95:
                reasons.append("excellent discriminative performance")
            elif row['ROC-AUC'] >= 0.85:
                reasons.append("good discriminative ability")
        
        if 'Sensitivity' in row and pd.notna(row['Sensitivity']) and row['Sensitivity'] >= 0.9:
            reasons.append("high sensitivity")
        
        if 'Specificity' in row and pd.notna(row['Specificity']) and row['Specificity'] >= 0.9:
            reasons.append("high specificity")
        
        if 'F1-Score' in row and pd.notna(row['F1-Score']) and row['F1-Score'] >= 0.9:
            reasons.append("well-balanced precision and recall")
        
        if row.get('Status') == 'PASS':
            reasons.append("passes all tests")
        
        return ", ".join(reasons) if reasons else "meets basic requirements"
    
    def _detailed_model_comparison(self, ranking):
        """Provide detailed comparison between top models"""
        print(f"\nDETAILED COMPARISON:")
        
        if len(ranking) > 1:
            best = ranking[0]
            second = ranking[1]
            
            score_diff = best['score'] - second['score']
            print(f"   Best model ({best['model']}) outperforms runner-up by {score_diff:.4f} points")
            
            if score_diff < 0.05:
                print("   [WARNING] Performance difference is marginal - consider other factors:")
                print("      - Model complexity and interpretability")
                print("      - Training time and computational requirements")
                print("      - Robustness across different datasets")
    
    def _provide_comparison_recommendations(self, df, ranking):
        """Provide recommendations based on model comparison"""
        print(f"\nCOMPARISON INSIGHTS:")
        
        if len(ranking) >= 2:
            best_model = ranking[0]['model']
            print(f"   - Recommended: {best_model}")
            print(f"   - Reason: {ranking[0]['reason']}")
            
            # Check for concerning patterns
            all_grades = [row.get('Grade', 'Unknown') for _, row in df.iterrows()]
            if all(grade in ['POOR', 'FAIR'] for grade in all_grades):
                print(f"   [WARNING] All models show suboptimal performance - consider:")
                print(f"      - Data quality improvement")
                print(f"      - Feature engineering")
                print(f"      - Different algorithms or ensemble methods")
    
    # Interpretation helper methods
    def _interpret_accuracy(self, acc):
        if acc >= 0.95: return "Excellent"
        elif acc >= 0.90: return "Very Good"
        elif acc >= 0.80: return "Good"
        elif acc >= 0.70: return "Fair"
        else: return "Poor"
    
    def _interpret_auc(self, auc):
        if auc >= 0.95: return "Excellent discrimination"
        elif auc >= 0.90: return "Very good discrimination"
        elif auc >= 0.80: return "Good discrimination"
        elif auc >= 0.70: return "Fair discrimination"
        else: return "Poor discrimination"
    
    def _interpret_f1(self, f1):
        if f1 >= 0.90: return "Excellent balance"
        elif f1 >= 0.80: return "Good balance"
        elif f1 >= 0.70: return "Fair balance"
        else: return "Poor balance"
    
    def _interpret_precision(self, prec):
        if prec >= 0.90: return "Very low false positives"
        elif prec >= 0.80: return "Low false positives"
        elif prec >= 0.70: return "Moderate false positives"
        else: return "High false positives"
    
    def _interpret_recall(self, rec):
        if rec >= 0.90: return "Very low false negatives"
        elif rec >= 0.80: return "Low false negatives"
        elif rec >= 0.70: return "Moderate false negatives"
        else: return "High false negatives"
    
    def _interpret_sensitivity(self, sens):
        if sens >= 0.95: return "Excellent for screening"
        elif sens >= 0.90: return "Very good for screening"
        elif sens >= 0.80: return "Good for screening"
        else: return "May miss cases"
    
    def _interpret_specificity(self, spec):
        if spec >= 0.95: return "Excellent - very few false alarms"
        elif spec >= 0.90: return "Very good - low false alarms"
        elif spec >= 0.80: return "Good - moderate false alarms"
        else: return "High false alarm rate"
    
    def _interpret_ppv(self, ppv):
        if ppv >= 0.90: return "High confidence in positive predictions"
        elif ppv >= 0.80: return "Good confidence in positive predictions"
        else: return "Lower confidence in positive predictions"
    
    def _interpret_npv(self, npv):
        if npv >= 0.95: return "High confidence in negative predictions"
        elif npv >= 0.90: return "Good confidence in negative predictions"
        else: return "Lower confidence in negative predictions"
    
    def _interpret_fp(self, fp, total):
        rate = fp / total
        if rate < 0.05: return "Very low false alarm rate"
        elif rate < 0.10: return "Low false alarm rate"
        elif rate < 0.20: return "Moderate false alarm rate"
        else: return "High false alarm rate - concerning"
    
    def _interpret_fn(self, fn, total):
        rate = fn / total
        if rate < 0.05: return "Very low miss rate"
        elif rate < 0.10: return "Low miss rate"
        elif rate < 0.20: return "Moderate miss rate"
        else: return "High miss rate - concerning"

def main():
    """Main function to handle command line arguments and run analysis"""
    if len(sys.argv) < 2:
        print("Usage: python ml_evaluate.py <result_file1> [result_file2] [result_file3]")
        print("\nExamples:")
        print("  python ml_evaluate.py cancer_model_results.json")
        print("  python ml_evaluate.py model_v1.json model_v2.json model_v3.json")
        return
    
    if len(sys.argv) > 4:
        print("Maximum of 3 result files supported for comparison")
        return
    
    analyzer = MLResultsAnalyzer()
    
    # Single model analysis
    if len(sys.argv) == 2:
        result = analyzer.analyze_single_result(sys.argv[1])
        if result is None:
            sys.exit(1)
    
    # Multi-model comparison
    else:
        result_files = sys.argv[1:]
        analyzer.load_results(result_files)
        if len(analyzer.results) >= 2:
            analyzer.compare_results()
        elif len(analyzer.results) == 1:
            print("Only 1 valid file loaded, switching to single model analysis...\n")
            analyzer.analyze_single_result(analyzer.results[0]['file_path'])
        else:
            print("No valid result files found")
            sys.exit(1)

if __name__ == "__main__":
    main()