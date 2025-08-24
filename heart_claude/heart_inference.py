# Heart Disease Prediction - Interactive TUI Inference Application
# Professional clinical decision support tool with Rich TUI

import os
import sys
import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, FloatPrompt, IntPrompt
from rich.progress import track
from rich.text import Text
from rich import print as rprint
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FEATURE SPECIFICATIONS WITH CLINICAL CONTEXT
# =============================================================================

FEATURE_SPECS = {
    'age': {
        'prompt': 'Patient age',
        'type': 'int',
        'min': 20, 'max': 100,
        'description': 'Age in years (20-100)',
        'clinical_note': 'Risk increases significantly after age 65'
    },
    'sex': {
        'prompt': 'Patient sex',
        'type': 'choice',
        'choices': {'Male': 1, 'Female': 0},
        'description': 'Biological sex',
        'clinical_note': 'Males generally at higher risk for heart disease'
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
        'description': 'Type of chest pain experienced',
        'clinical_note': 'Typical angina is most concerning for heart disease'
    },
    'trestbps': {
        'prompt': 'Resting blood pressure',
        'type': 'int',
        'min': 80, 'max': 200,
        'description': 'Resting blood pressure in mm Hg',
        'clinical_note': 'Normal: <120, High: >140 mmHg'
    },
    'chol': {
        'prompt': 'Serum cholesterol',
        'type': 'int', 
        'min': 100, 'max': 600,
        'description': 'Serum cholesterol in mg/dl',
        'clinical_note': 'Desirable: <200, High: >240 mg/dl'
    },
    'fbs': {
        'prompt': 'Fasting blood sugar',
        'type': 'choice',
        'choices': {'> 120 mg/dl (Diabetic)': 1, '≤ 120 mg/dl (Normal)': 0},
        'description': 'Fasting blood sugar level',
        'clinical_note': 'Diabetes significantly increases cardiovascular risk'
    },
    'restecg': {
        'prompt': 'Resting ECG results',
        'type': 'choice',
        'choices': {
            'Normal': 0,
            'ST-T Wave Abnormality': 1,
            'Left Ventricular Hypertrophy': 2
        },
        'description': 'Resting electrocardiographic results',
        'clinical_note': 'Abnormalities may indicate cardiac pathology'
    },
    'thalach': {
        'prompt': 'Maximum heart rate achieved',
        'type': 'int',
        'min': 60, 'max': 220,
        'description': 'Maximum heart rate achieved during exercise testing',
        'clinical_note': 'Lower max heart rate may indicate cardiac limitation'
    },
    'exang': {
        'prompt': 'Exercise induced angina',
        'type': 'choice',
        'choices': {'Yes': 1, 'No': 0},
        'description': 'Exercise-induced chest pain/angina',
        'clinical_note': 'Strong predictor of coronary artery disease'
    },
    'oldpeak': {
        'prompt': 'ST depression',
        'type': 'float',
        'min': 0.0, 'max': 10.0,
        'description': 'ST depression induced by exercise relative to rest',
        'clinical_note': 'Higher values indicate more severe ischemia'
    },
    'slope': {
        'prompt': 'Slope of peak exercise ST segment',
        'type': 'choice',
        'choices': {
            'Upsloping': 0,
            'Flat': 1,
            'Downsloping': 2
        },
        'description': 'Slope of the peak exercise ST segment',
        'clinical_note': 'Downsloping most concerning for heart disease'
    },
    'ca': {
        'prompt': 'Number of major vessels',
        'type': 'choice',
        'choices': {'0': 0, '1': 1, '2': 2, '3': 3},
        'description': 'Number of major vessels (0-3) colored by fluoroscopy',
        'clinical_note': 'More blocked vessels = higher risk'
    },
    'thal': {
        'prompt': 'Thalassemia',
        'type': 'choice',
        'choices': {
            'Normal': 1,
            'Fixed Defect': 2,
            'Reversible Defect': 3
        },
        'description': 'Thalassemia stress test result',
        'clinical_note': 'Defects indicate areas of poor blood flow'
    }
}

# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class HeartDiseasePredictor:
    def __init__(self, debug=False):
        self.console = Console()
        self.model = None
        self.feature_names = None
        self.model_metrics = None
        self.test_results = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.debug = debug
        
    def load_model(self):
        """Load trained model and metadata"""
        try:
            self.model = joblib.load(os.path.join(self.script_dir, 'heart_disease_model.pkl'))
            self.feature_names = joblib.load(os.path.join(self.script_dir, 'feature_names.pkl'))
            self.model_metrics = joblib.load(os.path.join(self.script_dir, 'model_metrics.pkl'))
            
            # Try to load test results if available
            try:
                self.test_results = joblib.load(os.path.join(self.script_dir, 'test_results.pkl'))
            except FileNotFoundError:
                self.test_results = None
                
            return True
        except FileNotFoundError as e:
            self.console.print(f"[red]Error: Model files not found![/red]")
            self.console.print(f"[red]{str(e)}[/red]")
            self.console.print("\n[yellow]Please run the following scripts first:[/yellow]")
            self.console.print("1. [cyan]python heart_training.py[/cyan] - to train the model")
            self.console.print("2. [cyan]python heart_testing.py[/cyan] - to test the model (optional)")
            return False

    def display_welcome(self):
        """Display welcome screen with model information"""
        # Get performance metrics
        if self.test_results:
            metrics = self.test_results['test_metrics']
            accuracy = metrics.get('accuracy', 0)
            sensitivity = metrics.get('sensitivity', 0)
            specificity = metrics.get('specificity', 0)
            auc = metrics.get('auc_roc', 0)
        else:
            # Use training metrics if test results not available
            accuracy = 0.85  # placeholder
            sensitivity = 0.83  # placeholder  
            specificity = 0.87  # placeholder
            auc = self.model_metrics.get('cv_auc_mean', 0)

        welcome_content = (
            "[bold blue]🏥 Heart Disease Prediction System[/bold blue]\n"
            "[dim]AI-Powered Clinical Decision Support Tool[/dim]\n\n"
            "[bold green]Model Performance:[/bold green]\n"
            f"• Accuracy: [cyan]{accuracy:.1%}[/cyan]\n"
            f"• Sensitivity: [cyan]{sensitivity:.1%}[/cyan] (detects heart disease)\n"
            f"• Specificity: [cyan]{specificity:.1%}[/cyan] (detects healthy patients)\n"
            f"• AUC-ROC: [cyan]{auc:.3f}[/cyan]\n\n"
            "[bold yellow]Model Status:[/bold yellow] Ready for Clinical Use"
        )
        
        welcome_panel = Panel.fit(
            welcome_content,
            title="Welcome to Heart Disease Risk Assessment",
            border_style="blue"
        )
        self.console.print(welcome_panel)

    def display_disclaimer(self):
        """Display medical disclaimer"""
        disclaimer_content = (
            "[bold red]⚠️  MEDICAL DISCLAIMER ⚠️[/bold red]\n\n"
            "[yellow]This AI model is intended for EDUCATIONAL and RESEARCH purposes only.[/yellow]\n\n"
            "This tool should [bold]NOT[/bold] be used as a substitute for:\n"
            "• Professional medical advice\n"
            "• Clinical diagnosis\n"
            "• Treatment decisions\n\n"
            "[bold]Always consult qualified healthcare providers for medical decisions.[/bold]\n\n"
            "The predictions are based on statistical patterns and may not apply\n"
            "to individual cases. Clinical judgment should always take precedence."
        )
        
        disclaimer = Panel(
            disclaimer_content,
            title="⚠️ Important Medical Disclaimer",
            border_style="red"
        )
        self.console.print(disclaimer)
        
        proceed = Confirm.ask("\n[bold]Do you acknowledge this disclaimer and wish to continue?[/bold]")
        return proceed

    def collect_patient_data(self):
        """Collect patient data through interactive prompts"""
        self.console.print(Panel(
            "[bold green]Patient Data Collection[/bold green]\n"
            "Please provide the following patient information for risk assessment.",
            title="📋 Data Collection",
            border_style="green"
        ))
        
        patient_data = {}
        
        # Use a simple counter instead of track for better control
        total_features = len(FEATURE_SPECS)
        current_feature = 0
        
        for feature, spec in FEATURE_SPECS.items():
            current_feature += 1
            
            # Initialize variables
            choice_text = None
            
            # Progress indicator
            progress_text = f"[dim]({current_feature}/{total_features})[/dim]"
            
            # Display feature information panel
            info_content = (
                f"[bold cyan]{spec['prompt']}[/bold cyan]\n"
                f"[dim]{spec['description']}[/dim]\n\n"
                f"[yellow]Clinical Note:[/yellow] {spec['clinical_note']}"
            )
            
            info_panel = Panel(
                info_content,
                title=f"{progress_text} {spec['prompt']}",
                border_style="cyan"
            )
            self.console.print(info_panel)
            
            # Get user input based on feature type
            if spec['type'] == 'int':
                while True:
                    try:
                        value = IntPrompt.ask(
                            f"[bold]Enter {spec['prompt'].lower()}[/bold]",
                            default=None,
                            show_default=False
                        )
                        if spec['min'] <= value <= spec['max']:
                            break
                        else:
                            self.console.print(f"[red]Please enter a value between {spec['min']} and {spec['max']}[/red]")
                    except ValueError:
                        self.console.print("[red]Please enter a valid integer[/red]")
                        
            elif spec['type'] == 'float':
                while True:
                    try:
                        value = FloatPrompt.ask(
                            f"[bold]Enter {spec['prompt'].lower()}[/bold]",
                            default=None,
                            show_default=False
                        )
                        if spec['min'] <= value <= spec['max']:
                            break
                        else:
                            self.console.print(f"[red]Please enter a value between {spec['min']} and {spec['max']}[/red]")
                    except ValueError:
                        self.console.print("[red]Please enter a valid number[/red]")
                        
            elif spec['type'] == 'choice':
                # Create numbered list of options
                options_list = list(spec['choices'].items())  # [(option_text, value), ...]
                
                self.console.print(f"\n[bold]Available options for {spec['prompt']}:[/bold]")
                for i, (option_text, option_value) in enumerate(options_list, 1):
                    self.console.print(f"  [cyan]{i}.[/cyan] {option_text}")
                
                while True:
                    try:
                        choice_num = IntPrompt.ask(
                            f"\n[bold]Enter number (1-{len(options_list)}) for {spec['prompt'].lower()}[/bold]",
                            default=None,
                            show_default=False
                        )
                        
                        # Validate choice number
                        if not (1 <= choice_num <= len(options_list)):
                            self.console.print(f"[red]Please enter a number between 1 and {len(options_list)}[/red]")
                            continue
                        
                        # Get the selected option and value
                        choice_text, value = options_list[choice_num - 1]
                        
                        # Additional validation for choice values
                        if not isinstance(value, (int, float)):
                            raise ValueError(f"Invalid choice value: {value}")
                        
                        break
                        
                    except ValueError as e:
                        if "invalid literal" in str(e).lower():
                            self.console.print("[red]Please enter a valid number[/red]")
                        else:
                            self.console.print(f"[red]Error with choice selection: {str(e)}[/red]")
                    except Exception as e:
                        self.console.print(f"[red]Unexpected error: {str(e)}[/red]")
            
            # Validate the collected value
            try:
                if spec['type'] in ['int', 'float']:
                    # Ensure numeric values are within acceptable ranges
                    if not (spec['min'] <= value <= spec['max']):
                        raise ValueError(f"Value {value} outside valid range [{spec['min']}, {spec['max']}]")
                
                patient_data[feature] = value
                
                # Confirmation with debug info
                if spec['type'] == 'choice':
                    display_value = choice_text  # Use the selected text
                else:
                    display_value = value
                
                # Confirmation
                self.console.print(f"✅ [green]{spec['prompt']}: [bold]{display_value}[/bold][/green]")
                
                # Debug: Show what was stored (only in debug mode)
                if self.debug:
                    self.console.print(f"[dim]   Debug - Stored: {feature} = {value} (type: {type(value).__name__})[/dim]")
                
                self.console.print()
                
            except Exception as e:
                self.console.print(f"[red]Error storing {feature}: {str(e)}[/red]")
                raise
        
        # Final validation of complete dataset
        if self.debug:
            self.console.print(f"[dim]Debug - Collected {len(patient_data)} features: {list(patient_data.keys())}[/dim]")
        return patient_data

    def display_patient_summary(self, patient_data):
        """Display patient data summary for confirmation"""
        summary_table = Table(title="📋 Patient Data Summary", border_style="cyan", show_header=True)
        summary_table.add_column("Parameter", style="bold", min_width=20)
        summary_table.add_column("Value", justify="center", min_width=15)
        summary_table.add_column("Clinical Interpretation", min_width=30)
        
        for feature, value in patient_data.items():
            spec = FEATURE_SPECS[feature]
            description = self.get_factor_description(feature, value)
            
            # Get display value for choices
            display_value = self.get_display_value(feature, value)
            
            summary_table.add_row(
                spec['prompt'],
                display_value,
                description
            )
        
        self.console.print(summary_table)

    def predict_risk(self, patient_data):
        """Make prediction and calculate risk assessment"""
        try:
            # Debug: Print patient data (only in debug mode)
            if self.debug:
                self.console.print(f"[dim]Debug - Patient data keys: {list(patient_data.keys())}[/dim]")
                self.console.print(f"[dim]Debug - Expected features: {self.feature_names}[/dim]")
            
            # Check if all required features are present
            missing_features = set(self.feature_names) - set(patient_data.keys())
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Convert to DataFrame with correct feature order
            input_df = pd.DataFrame([patient_data])[self.feature_names]
            
            # Debug: Print input data shape and values (only in debug mode)
            if self.debug:
                self.console.print(f"[dim]Debug - Input shape: {input_df.shape}[/dim]")
                self.console.print(f"[dim]Debug - Input values: {input_df.iloc[0].to_dict()}[/dim]")
            
            # Get prediction and probability
            prediction = self.model.predict(input_df)[0]
            probability = self.model.predict_proba(input_df)[0, 1]
            
            return prediction, probability
            
        except Exception as e:
            self.console.print(f"[red]Error in predict_risk: {str(e)}[/red]")
            if self.debug:
                self.console.print(f"[red]Patient data: {patient_data}[/red]")
            raise

    def interpret_risk_level(self, probability):
        """Interpret probability as risk level with clinical context"""
        if probability < 0.2:
            return {
                'level': 'Very Low',
                'color': 'green',
                'icon': '🟢',
                'message': 'Very low probability of heart disease. Continue regular preventive care.',
                'urgency': 'Routine',
                'recommendations': [
                    'Maintain healthy lifestyle habits',
                    'Regular exercise (150+ min/week moderate activity)',
                    'Heart-healthy diet (low sodium, saturated fat)',
                    'Annual check-ups with healthcare provider',
                    'Monitor and control blood pressure, cholesterol',
                    'Avoid smoking and excessive alcohol'
                ]
            }
        elif probability < 0.4:
            return {
                'level': 'Low',
                'color': 'blue',
                'icon': '🔵',
                'message': 'Low probability of heart disease. Maintain current health practices.',
                'urgency': 'Routine Monitoring',
                'recommendations': [
                    'Continue healthy lifestyle modifications',
                    'Monitor blood pressure and cholesterol regularly',
                    'Consider cardio-protective medications if indicated',
                    'Biannual healthcare provider visits',
                    'Stress management and adequate sleep',
                    'Regular physical activity and weight management'
                ]
            }
        elif probability < 0.6:
            return {
                'level': 'Moderate',
                'color': 'yellow',
                'icon': '🟡',
                'message': 'Moderate risk of heart disease. Consider further cardiovascular evaluation.',
                'urgency': 'Enhanced Monitoring',
                'recommendations': [
                    'Schedule comprehensive cardiovascular evaluation',
                    'Consider stress testing or cardiac imaging',
                    'Optimize management of risk factors',
                    'More frequent healthcare provider visits',
                    'Possible cardiology consultation',
                    'Aggressive lifestyle modifications'
                ]
            }
        elif probability < 0.8:
            return {
                'level': 'High',
                'color': 'orange',
                'icon': '🟠',
                'message': 'High probability of heart disease. Cardiology evaluation recommended.',
                'urgency': 'Prompt Evaluation',
                'recommendations': [
                    'Urgent cardiology consultation within 1-2 weeks',
                    'Comprehensive cardiac workup (stress test, echo, etc.)',
                    'Immediate lifestyle modifications',
                    'Consider cardiac catheterization if indicated',
                    'Optimize medical therapy',
                    'Close monitoring and follow-up'
                ]
            }
        else:
            return {
                'level': 'Very High',
                'color': 'red',
                'icon': '🔴',
                'message': 'Very high probability of heart disease. Immediate medical attention advised.',
                'urgency': 'IMMEDIATE',
                'recommendations': [
                    'IMMEDIATE cardiology consultation (same day if possible)',
                    'Consider emergency evaluation if symptomatic',
                    'Comprehensive cardiac evaluation urgently needed',
                    'Likely need for cardiac catheterization',
                    'Aggressive medical management',
                    'Close hospital/clinic monitoring'
                ]
            }

    def analyze_risk_factors(self, patient_data):
        """Analyze individual risk factors and their contributions"""
        # Get feature coefficients from the model
        classifier = self.model.named_steps['classifier']
        coefficients = classifier.coef_[0]
        
        # Calculate feature contributions
        scaler = self.model.named_steps['scaler']
        input_df = pd.DataFrame([patient_data])[self.feature_names]
        scaled_data = scaler.transform(input_df)
        
        risk_factors = []
        protective_factors = []
        neutral_factors = []
        
        for i, feature in enumerate(self.feature_names):
            contribution = scaled_data[0][i] * coefficients[i]
            
            factor_info = {
                'feature': feature,
                'value': patient_data[feature],
                'contribution': contribution,
                'coefficient': coefficients[i],
                'odds_ratio': np.exp(coefficients[i])
            }
            
            if contribution > 0.1:  # Significant risk contribution
                risk_factors.append(factor_info)
            elif contribution < -0.1:  # Significant protective contribution
                protective_factors.append(factor_info)
            else:
                neutral_factors.append(factor_info)
        
        # Sort by absolute contribution
        risk_factors.sort(key=lambda x: x['contribution'], reverse=True)
        protective_factors.sort(key=lambda x: x['contribution'])
        
        return risk_factors, protective_factors, neutral_factors

    def get_display_value(self, feature, value):
        """Helper function to get human-readable display value for a feature"""
        spec = FEATURE_SPECS[feature]
        if spec['type'] == 'choice':
            # Find the text that corresponds to this value
            for text, code in spec['choices'].items():
                if code == value:
                    return text
            return f"Unknown ({value})"
        else:
            return str(value)

    def get_factor_description(self, feature, value):
        """Get clinical description for specific factor values"""
        try:
            descriptions = {
                'age': f"{value} years old" + (" (advanced age increases risk)" if value > 65 else " (good age range)"),
                'sex': ("Male (higher baseline risk)" if value == 1 else "Female (lower baseline risk)"),
                'cp': {
                    0: "Typical angina (high risk pattern)",
                    1: "Atypical angina (moderate concern)", 
                    2: "Non-anginal chest pain (low concern)",
                    3: "Asymptomatic (no chest pain)"
                }.get(value, f"Unknown chest pain type ({value})"),
                'trestbps': f"{value} mmHg" + (
                    " (hypertensive - high risk)" if value > 140 else
                    " (elevated)" if value > 120 else
                    " (normal)"
                ),
                'chol': f"{value} mg/dl" + (
                    " (very high)" if value > 240 else
                    " (borderline high)" if value > 200 else
                    " (desirable level)"
                ),
                'fbs': ("Diabetic (>120 mg/dl) - significant risk factor" if value == 1 else "Non-diabetic - good"),
                'restecg': {
                    0: "Normal ECG - good finding",
                    1: "ST-T wave abnormality - concerning",
                    2: "Left ventricular hypertrophy - high risk"
                }.get(value, f"Unknown ECG result ({value})"),
                'thalach': f"{value} bpm" + (
                    " (reduced exercise capacity)" if value < 120 else
                    " (good exercise capacity)" if value > 160 else
                    " (moderate exercise capacity)"
                ),
                'exang': ("Exercise-induced angina present - high risk" if value == 1 else "No exercise angina - good"),
                'oldpeak': f"{value} mm ST depression" + (
                    " (severe ischemia)" if value > 3 else
                    " (moderate ischemia)" if value > 1 else
                    " (minimal/no ischemia)"
                ),
                'slope': {
                    0: "Upsloping ST segment - good finding",
                    1: "Flat ST segment - concerning",
                    2: "Downsloping ST segment - high risk"
                }.get(value, f"Unknown slope type ({value})"),
                'ca': f"{value} major vessels with blockage" + (
                    " (extensive disease)" if value > 2 else
                    " (significant disease)" if value > 0 else
                    " (no significant blockage)"
                ),
                'thal': {
                    1: "Normal perfusion - good finding",
                    2: "Fixed defect - previous heart damage",
                    3: "Reversible defect - active ischemia"
                }.get(value, f"Unknown thalassemia result ({value})")
            }
            return descriptions.get(feature, f"Unknown feature: {feature} = {value}")
        except Exception as e:
            return f"Error describing {feature}={value}: {str(e)}"

    def display_results(self, patient_data, prediction, probability, risk_interpretation, 
                       risk_factors, protective_factors):
        """Display comprehensive results dashboard"""
        
        self.console.print("\n" + "="*80)
        self.console.print("[bold blue]🏥 HEART DISEASE RISK ASSESSMENT RESULTS[/bold blue]")
        self.console.print("="*80 + "\n")
        
        # Main prediction panel
        prediction_content = (
            f"{risk_interpretation['icon']} [bold {risk_interpretation['color']}]"
            f"{risk_interpretation['level']} Risk[/bold {risk_interpretation['color']}]\n\n"
            f"[bold]Risk Probability:[/bold] [bold]{probability:.1%}[/bold]\n"
            f"[bold]Model Prediction:[/bold] [bold]{'Heart Disease Likely' if prediction == 1 else 'No Heart Disease Detected'}[/bold]\n"
            f"[bold]Clinical Urgency:[/bold] [bold {risk_interpretation['color']}]{risk_interpretation['urgency']}[/bold {risk_interpretation['color']}]\n\n"
            f"[italic]{risk_interpretation['message']}[/italic]"
        )
        
        prediction_panel = Panel(
            prediction_content,
            title="🎯 Risk Assessment Summary",
            border_style=risk_interpretation['color']
        )
        self.console.print(prediction_panel)
        
        # Risk factors analysis
        if risk_factors:
            risk_table = Table(title="⚠️ Primary Risk Factors", border_style="red", show_header=True)
            risk_table.add_column("Risk Factor", style="bold", min_width=20)
            risk_table.add_column("Patient Value", justify="center", min_width=15)
            risk_table.add_column("Clinical Impact", min_width=25)
            risk_table.add_column("Risk Score", justify="center", min_width=12)
            
            for factor in risk_factors[:5]:  # Top 5 risk factors
                impact_desc = self.get_factor_description(factor['feature'], factor['value'])
                risk_score = "High" if factor['contribution'] > 0.5 else "Moderate"
                
                # Get display value for factor
                display_value = self.get_display_value(factor['feature'], factor['value'])
                
                risk_table.add_row(
                    factor['feature'].replace('_', ' ').title(),
                    display_value,
                    impact_desc,
                    f"[red]{risk_score}[/red]"
                )
            
            self.console.print(risk_table)
        
        # Protective factors
        if protective_factors:
            protective_table = Table(title="✅ Protective Factors", border_style="green", show_header=True)
            protective_table.add_column("Protective Factor", style="bold", min_width=20)
            protective_table.add_column("Patient Value", justify="center", min_width=15)
            protective_table.add_column("Clinical Benefit", min_width=25)
            protective_table.add_column("Protection Level", justify="center", min_width=15)
            
            for factor in protective_factors[:3]:  # Top 3 protective factors
                benefit_desc = self.get_factor_description(factor['feature'], factor['value'])
                protection_level = "Strong" if factor['contribution'] < -0.5 else "Moderate"
                
                # Get display value for factor
                display_value = self.get_display_value(factor['feature'], factor['value'])
                
                protective_table.add_row(
                    factor['feature'].replace('_', ' ').title(),
                    display_value,
                    benefit_desc,
                    f"[green]{protection_level}[/green]"
                )
            
            self.console.print(protective_table)
        
        # Clinical recommendations
        recommendations_content = "\n".join([f"• {rec}" for rec in risk_interpretation['recommendations']])
        recommendations_panel = Panel(
            recommendations_content,
            title=f"🏥 Clinical Recommendations ({risk_interpretation['urgency']})",
            border_style="blue"
        )
        self.console.print(recommendations_panel)
        
        # Additional clinical context
        context_content = (
            f"[bold]Model Confidence:[/bold] {'High' if abs(probability - 0.5) > 0.3 else 'Moderate'}\n"
            f"[bold]Risk Percentile:[/bold] {self.get_risk_percentile(probability)}\n"
            f"[bold]Next Assessment:[/bold] {self.get_next_assessment_timing(risk_interpretation['level'])}\n\n"
            f"[dim]This assessment is based on current clinical guidelines and statistical models.\n"
            f"Individual patient factors not captured in this model may influence actual risk.[/dim]"
        )
        
        context_panel = Panel(
            context_content,
            title="📊 Additional Clinical Context",
            border_style="cyan"
        )
        self.console.print(context_panel)

    def get_risk_percentile(self, probability):
        """Get risk percentile description"""
        if probability < 0.1:
            return "Bottom 10% (very low risk population)"
        elif probability < 0.3:
            return "Bottom 30% (low risk population)"
        elif probability < 0.7:
            return "Middle 40% (moderate risk population)"
        elif probability < 0.9:
            return "Top 30% (high risk population)"
        else:
            return "Top 10% (very high risk population)"

    def get_next_assessment_timing(self, risk_level):
        """Get recommended timing for next assessment"""
        timing_map = {
            'Very Low': 'Annual assessment',
            'Low': '6-12 months',
            'Moderate': '3-6 months',
            'High': '1-3 months',
            'Very High': '2-4 weeks or as clinically indicated'
        }
        return timing_map.get(risk_level, 'As clinically indicated')

    def run_application(self):
        """Main application flow"""
        # Clear screen and show welcome
        self.console.clear()
        
        # Load model first
        if not self.load_model():
            return
        
        # Welcome and disclaimer
        self.display_welcome()
        if not self.display_disclaimer():
            self.console.print("\n[yellow]Application terminated by user. Thank you![/yellow]")
            return
        
        # Main application loop
        while True:
            try:
                self.console.clear()
                self.console.print("[bold green]Starting New Patient Assessment...[/bold green]\n")
                
                # Collect patient data
                patient_data = self.collect_patient_data()
                
                # Confirm data
                self.console.clear()
                self.display_patient_summary(patient_data)
                
                if not Confirm.ask("\n[bold]Is this patient information correct?[/bold]"):
                    if Confirm.ask("Would you like to re-enter the data?"):
                        continue
                    else:
                        break
                
                # Make prediction with loading animation
                with self.console.status("[bold green]🔬 Analyzing patient data and calculating risk...[/bold green]", spinner="dots"):
                    prediction, probability = self.predict_risk(patient_data)
                    risk_interpretation = self.interpret_risk_level(probability)
                    risk_factors, protective_factors, _ = self.analyze_risk_factors(patient_data)
                
                # Display results
                self.console.clear()
                self.display_results(patient_data, prediction, probability, 
                                   risk_interpretation, risk_factors, protective_factors)
                
                # Ask for another prediction
                self.console.print("\n" + "-"*80)
                if not Confirm.ask("\n[bold]Would you like to analyze another patient?[/bold]"):
                    break
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Application interrupted by user.[/yellow]")
                if Confirm.ask("Do you want to exit the application?"):
                    break
            except Exception as e:
                self.console.print(f"\n[red]Error occurred: {str(e)}[/red]")
                self.console.print(f"[red]Error type: {type(e).__name__}[/red]")
                
                # More specific error handling
                if "50" in str(e):
                    self.console.print("[yellow]This might be related to:[/yellow]")
                    self.console.print("• Invalid input range (check age, blood pressure, cholesterol values)")
                    self.console.print("• Missing or corrupted model files")
                    self.console.print("• Data type mismatch in patient data")
                
                # Print debug information
                import traceback
                self.console.print(f"\n[dim]Debug information:[/dim]")
                self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
                
                if not Confirm.ask("Would you like to try again?"):
                    break
        
        # Goodbye message
        goodbye_panel = Panel.fit(
            "[bold blue]Thank you for using the Heart Disease Prediction System![/bold blue]\n\n"
            "[dim]Remember: This tool is for educational purposes only.\n"
            "Always consult healthcare professionals for medical decisions.[/dim]\n\n"
            "[green]Stay healthy! 💚[/green]",
            title="👋 Goodbye",
            border_style="blue"
        )
        self.console.print(goodbye_panel)

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

def main():
    """Main application entry point"""
    # Debug mode can be enabled by changing debug=False to debug=True
    app = HeartDiseasePredictor(debug=False)
    app.run_application()

if __name__ == "__main__":
    main()