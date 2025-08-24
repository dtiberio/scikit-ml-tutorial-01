# SPDX-License-Identifier: MIT
# Copyright © 2025 github.com/dtiberio

"""
Breast Cancer Prediction TUI Application

Interactive clinical decision support tool for breast cancer diagnosis prediction
using the trained ML model with professional medical interface.

Author: Claude Code
Version: 1.0
"""

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
import warnings
warnings.filterwarnings('ignore')

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
            model_dir = os.path.join(script_dir, 'models')

            model_path = os.path.join(model_dir, 'breast_cancer_model.pkl')
            feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
            metrics_path = os.path.join(model_dir, 'model_metrics.pkl')

            self.model = joblib.load(model_path)
            self.feature_names = joblib.load(feature_names_path)
            self.model_metrics = joblib.load(metrics_path)

            # Try to load test results if available
            try:
                test_results_path = os.path.join(script_dir, 'tests', 'cancer_model_test_results.json')
                import json
                with open(test_results_path, 'r') as f:
                    self.test_results = json.load(f)
            except FileNotFoundError:
                self.test_results = None

            self.console.print("[SUCCESS] Model loaded successfully!")
            return True
        except FileNotFoundError as e:
            self.console.print(f"[red]Error: Model files not found! {e}[/red]")
            return False

    def display_welcome(self):
        """Display welcome screen with model information"""
        if self.test_results and 'tests' in self.test_results:
            accuracy = self.test_results['tests']['accuracy']['value']
            sensitivity = self.test_results['tests']['clinical_metrics']['sensitivity']
            specificity = self.test_results['tests']['clinical_metrics']['specificity']
            auc = self.test_results['tests']['roc_auc']['value']

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
            title="[MEDICAL] Welcome",
            border_style="blue"
        )
        self.console.print(welcome_panel)

    def get_yes_no_input(self, prompt_text):
        """Custom yes/no input with better visual feedback"""
        while True:
            try:
                self.console.print(f"[cyan]→ {prompt_text} [y/n]:[/cyan]", end=" ")
                user_input = input().lower().strip()
                
                # Show what was entered for confirmation
                self.console.print(f"[dim]   You entered: {user_input}[/dim]")
                
                if user_input in ['y', 'yes', '1', 'true']:
                    return True
                elif user_input in ['n', 'no', '0', 'false']:
                    return False
                else:
                    self.console.print("[red]   Error: Please enter 'y' for yes or 'n' for no[/red]")
                    continue
                    
            except KeyboardInterrupt:
                raise
            except EOFError:
                raise

    def display_disclaimer(self):
        """Display medical disclaimer"""
        disclaimer = Panel(
            "[yellow][WARNING] MEDICAL DISCLAIMER [WARNING][/yellow]\n\n"
            "This AI model is intended for educational and research purposes only.\n"
            "It should NOT be used as a substitute for professional medical diagnosis\n"
            "or pathological examination. Breast cancer diagnosis requires proper\n"
            "histopathological analysis by qualified pathologists.\n\n"
            "Always consult qualified healthcare providers for medical decisions.",
            border_style="yellow"
        )
        self.console.print(disclaimer)

        proceed = self.get_yes_no_input("Do you acknowledge this disclaimer and wish to continue?")
        return proceed

    def get_float_input(self, prompt_text, min_val, max_val):
        """Custom float input with better visual feedback"""
        while True:
            try:
                # Use regular input() for better terminal compatibility
                self.console.print(f"[cyan]→ {prompt_text}:[/cyan]", end=" ")
                user_input = input()
                
                # Show what was entered for confirmation
                self.console.print(f"[dim]   You entered: {user_input}[/dim]")
                
                value = float(user_input)
                
                if value < min_val or value > max_val:
                    self.console.print(f"[red]   Error: Please enter a value between {min_val:.2f} and {max_val:.2f}[/red]")
                    continue
                    
                return value
                
            except ValueError:
                self.console.print(f"[red]   Error: Please enter a valid number[/red]")
                continue
            except KeyboardInterrupt:
                raise
            except EOFError:
                raise

    def collect_patient_data(self):
        """Collect tumor characteristics through interactive prompts"""
        self.console.print("\n[bold green]Tumor Characteristics Collection[/bold green]")
        self.console.print("Please provide the following tumor measurements:\n")

        patient_data = {}

        # For this demo, collect only the 10 mean features
        for i, (feature, spec) in enumerate(FEATURE_SPECS.items()):
            self.console.print(f"\n[bold blue]Feature {i+1}/10[/bold blue]")
            
            # Display feature information
            info_panel = Panel(
                f"[bold]{spec['prompt']}[/bold]\n{spec['description']}\n"
                f"[dim]Valid range: {spec['min']:.2f} - {spec['max']:.2f}[/dim]",
                border_style="cyan"
            )
            self.console.print(info_panel)

            # Get user input with better visual feedback
            value = self.get_float_input(
                f"Enter {spec['prompt'].lower()}", 
                spec['min'], 
                spec['max']
            )

            patient_data[feature] = value
            self.console.print(f"[SUCCESS] {spec['prompt']}: [green]{value:.3f}[/green] ✓\n")

        # For missing features, use dataset median values or ask user to specify
        self.console.print("[yellow]Note: Using dataset averages for remaining features not collected.[/yellow]")

        return patient_data

    def fill_missing_features(self, patient_data):
        """Fill missing features with dataset medians or reasonable defaults"""
        # Default values based on dataset statistics
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
                'icon': '[SUCCESS]',
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
                'icon': '[INFO]',
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
                'icon': '[WARNING]',
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
                'icon': '[ALERT]',
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
                'icon': '[CRITICAL]',
                'message': 'Very high probability of malignancy. Immediate comprehensive evaluation required.',
                'recommendations': [
                    'IMMEDIATE: Complete oncological evaluation',
                    'Rapid pathological diagnosis and staging',
                    'Urgent multidisciplinary tumor board review',
                    'Prepare for potential treatment planning'
                ]
            }

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
            title="[MICROSCOPE] Tumor Analysis Results",
            border_style=risk_interpretation['color']
        )
        self.console.print(prediction_panel)

        # Malignancy risk factors
        if malignancy_factors:
            risk_table = Table(title="[WARNING] Primary Malignancy Indicators", border_style="red")
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
            benign_table = Table(title="[SUCCESS] Benign Characteristics", border_style="green")
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
            title="[MEDICAL] Clinical Recommendations",
            border_style="blue"
        )
        self.console.print(recommendations_panel)

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
                if not self.get_yes_no_input("Are these tumor characteristics correct?"):
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
                if not self.get_yes_no_input("\nWould you like to analyze another tumor sample?"):
                    break

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Application interrupted by user.[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error occurred: {str(e)}[/red]")
                if not self.get_yes_no_input("Would you like to try again?"):
                    break

        self.console.print("\n[blue]Thank you for using the Breast Cancer Prediction System![/blue]")

if __name__ == "__main__":
    app = BreastCancerPredictor()

    if app.model is None:
        console = Console()
        console.print("[red]Failed to load model. Please ensure model files exist.[/red]")
        console.print("Run cancer_training.py first to train the model.")
        exit(1)

    app.run_application()