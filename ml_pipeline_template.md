# ML Pipeline Template for Supervised Learning

A comprehensive template for building supervised machine learning pipelines using scikit-learn. This template provides a standardized approach for data analysis, model training, testing, and deployment across regression and classification tasks.

## Template Applicability

### Supported Model Types

**Classification Models:**
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gradient Boosting Classifier
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Neural Networks (MLPClassifier)
- Naive Bayes

**Regression Models:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Support Vector Regression (SVR)
- Gradient Boosting Regressor
- Decision Tree Regressor
- K-Nearest Neighbors Regressor
- Neural Networks (MLPRegressor)

### Dataset Requirements
- **Size:** 100+ samples (preferably 1000+)
- **Format:** CSV, structured tabular data
- **Features:** Numeric and categorical features supported
- **Target:** Single target variable (binary/multiclass for classification, continuous for regression)

---

## Phase 1: Dataset Analysis and Model Recommendation (Steps 1-4)

This phase analyzes your dataset comprehensively and recommends the best ML algorithms based on data characteristics.

### Phase 1 Step Summary:
1. **Initial Dataset Review**: Load data, check memory usage, and basic information
2. **Comprehensive Data Analysis**: Assess data quality, feature types, and target variables 
3. **Model Recommendation Engine**: Smart algorithm selection based on dataset characteristics
4. **Dataset Summary Report**: Generate comprehensive analysis report with next steps

### Step 1: Initial Dataset Review

**Purpose:** Load dataset and understand basic characteristics to inform preprocessing decisions.

```python
# Step 1: Initial Dataset Review - Load and inspect basic dataset properties
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load your dataset - replace 'your_dataset.csv' with actual file path
df = pd.read_csv('your_dataset.csv')

# Basic dataset information - crucial for understanding data scope and complexity
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Quick data preview to understand structure
print("\nFirst 5 rows:")
print(df.head())

# Data types overview - helps identify numeric vs categorical features
print("\nData types:")
print(df.dtypes.value_counts())
```

### Step 2: Comprehensive Data Analysis

**Purpose:** Perform deep analysis of data quality, feature types, and identify potential target variables.

```python
# Step 2: Comprehensive Data Analysis - Deep dive into data characteristics
def analyze_dataset(df):
    """
    Comprehensive dataset analysis with model recommendations
    
    This function systematically analyzes:
    - Dataset basic information (shape, memory, types)
    - Data quality issues (missing values, duplicates)
    - Feature characteristics (numeric vs categorical)
    - Potential target variables (classification vs regression)
    """
    analysis = {
        'dataset_info': {},
        'data_quality': {},
        'feature_analysis': {},
        'target_analysis': {},
        'recommendations': {}
    }
    
    # Basic dataset info - foundation for all subsequent decisions
    analysis['dataset_info'] = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'dtypes': df.dtypes.value_counts().to_dict()
    }
    
    # Data quality assessment - critical for preprocessing planning
    missing_data = df.isnull().sum()
    duplicates = df.duplicated().sum()
    
    analysis['data_quality'] = {
        'missing_values': missing_data[missing_data > 0].to_dict(),
        'missing_percentage': (missing_data / len(df) * 100)[missing_data > 0].to_dict(),
        'duplicate_rows': duplicates,
        'duplicate_percentage': duplicates / len(df) * 100
    }
    
    # Feature analysis - determines preprocessing strategy
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    analysis['feature_analysis'] = {
        'numeric_features': len(numeric_cols),
        'categorical_features': len(categorical_cols),
        # High cardinality features may need special encoding (e.g., target encoding)
        'high_cardinality_features': [col for col in categorical_cols if df[col].nunique() > 20],
        # Sample numeric ranges help identify scaling needs
        'numeric_feature_ranges': {col: {'min': df[col].min(), 'max': df[col].max()} 
                                 for col in numeric_cols[:10]}  # Limit for readability
    }
    
    # Identify potential target columns - guides problem type selection
    potential_targets = []
    for col in df.columns:
        unique_values = df[col].nunique()
        if unique_values == 2:
            # Binary classification candidate
            potential_targets.append({'column': col, 'type': 'binary_classification', 'classes': unique_values})
        elif 2 < unique_values <= 10 and df[col].dtype in ['object', 'category', 'int64']:
            # Multiclass classification candidate
            potential_targets.append({'column': col, 'type': 'multiclass_classification', 'classes': unique_values})
        elif df[col].dtype in [np.number] and unique_values > 20:
            # Regression candidate (continuous target)
            potential_targets.append({'column': col, 'type': 'regression', 'unique_values': unique_values})
    
    analysis['target_analysis'] = {'potential_targets': potential_targets}
    
    return analysis

# Perform comprehensive analysis
dataset_analysis = analyze_dataset(df)

# Display analysis results - helps developers understand their data
print("=== DATASET ANALYSIS REPORT ===")
print(f"Dataset Shape: {dataset_analysis['dataset_info']['shape']}")
print(f"Memory Usage: {dataset_analysis['dataset_info']['memory_usage_mb']:.2f} MB")
print(f"Numeric Features: {dataset_analysis['feature_analysis']['numeric_features']}")
print(f"Categorical Features: {dataset_analysis['feature_analysis']['categorical_features']}")

# Data quality warnings - alerts to preprocessing needs
if dataset_analysis['data_quality']['missing_values']:
    print(f"⚠️  Missing Values: {len(dataset_analysis['data_quality']['missing_values'])} columns need imputation")
if dataset_analysis['data_quality']['duplicate_rows'] > 0:
    print(f"⚠️  Duplicate Rows: {dataset_analysis['data_quality']['duplicate_rows']} rows need deduplication")

# Potential target identification - guides problem formulation
print("\n🎯 Potential Target Variables:")
for target in dataset_analysis['target_analysis']['potential_targets']:
    print(f"  - {target['column']}: {target['type']}")
```

### Step 3: Model Recommendation Engine

**Purpose:** Intelligently recommend ML algorithms based on dataset characteristics and problem type.

```python
# Step 3: Model Recommendation Engine - Smart algorithm selection
def recommend_models(dataset_analysis, target_column, problem_type):
    """
    Recommend best models based on dataset characteristics
    
    This intelligent system considers:
    - Dataset size (small, medium, large)
    - Feature count and types
    - Data quality issues
    - Problem type (classification vs regression)
    """
    recommendations = {}
    
    # Extract key dataset characteristics for decision making
    n_samples = dataset_analysis['dataset_info']['shape'][0]
    n_features = dataset_analysis['dataset_info']['shape'][1] - 1  # Excluding target
    has_missing = len(dataset_analysis['data_quality']['missing_values']) > 0
    has_categorical = dataset_analysis['feature_analysis']['categorical_features'] > 0
    
    if problem_type == 'classification':
        recommendations['primary_models'] = []
        recommendations['alternative_models'] = []
        
        # Logistic Regression - interpretable baseline, works well with moderate features
        if n_features <= 100:
            recommendations['primary_models'].append({
                'model': 'LogisticRegression',
                'reason': 'Good baseline for moderate feature count, interpretable coefficients',
                'parameters': {'random_state': 42, 'max_iter': 1000}
            })
        
        # Random Forest - robust, handles mixed data types, provides feature importance
        recommendations['primary_models'].append({
            'model': 'RandomForestClassifier',
            'reason': 'Handles missing values, robust to outliers, provides feature importance',
            'parameters': {'random_state': 42, 'n_estimators': 100}
        })
        
        # Gradient Boosting - high performance for larger datasets
        if n_samples >= 1000:
            recommendations['primary_models'].append({
                'model': 'GradientBoostingClassifier',
                'reason': 'High performance for medium-large datasets, handles complex patterns',
                'parameters': {'random_state': 42, 'n_estimators': 100}
            })
        
        # SVM - effective for smaller datasets with good feature engineering
        if n_samples < 5000 and n_features < 100:
            recommendations['alternative_models'].append({
                'model': 'SVC',
                'reason': 'Effective for smaller datasets with moderate features, good generalization',
                'parameters': {'random_state': 42, 'probability': True}
            })
            
    elif problem_type == 'regression':
        recommendations['primary_models'] = []
        recommendations['alternative_models'] = []
        
        # Linear Regression - simple baseline, highly interpretable
        recommendations['primary_models'].append({
            'model': 'LinearRegression',
            'reason': 'Simple baseline, highly interpretable coefficients',
            'parameters': {}
        })
        
        # Ridge Regression - handles multicollinearity, prevents overfitting
        recommendations['primary_models'].append({
            'model': 'Ridge',
            'reason': 'Handles multicollinearity, prevents overfitting through L2 regularization',
            'parameters': {'random_state': 42}
        })
        
        # Random Forest - robust to outliers, non-linear relationships
        recommendations['primary_models'].append({
            'model': 'RandomForestRegressor',
            'reason': 'Handles missing values, captures non-linear relationships, robust',
            'parameters': {'random_state': 42, 'n_estimators': 100}
        })
        
        # Gradient Boosting - high performance for larger datasets
        if n_samples >= 1000:
            recommendations['primary_models'].append({
                'model': 'GradientBoostingRegressor',
                'reason': 'High performance for medium-large datasets, excellent predictive power',
                'parameters': {'random_state': 42, 'n_estimators': 100}
            })
    
    # Preprocessing recommendations based on data characteristics
    recommendations['preprocessing'] = []
    
    # Missing value handling
    if has_missing:
        recommendations['preprocessing'].append('Handle missing values with SimpleImputer (median for numeric, mode for categorical)')
    
    # Categorical encoding 
    if has_categorical:
        recommendations['preprocessing'].append('Encode categorical variables (OneHotEncoder for low cardinality, LabelEncoder for ordinal)')
    
    # Feature scaling for algorithms sensitive to feature magnitude
    if problem_type == 'classification' or any('Linear' in model['model'] or 'SV' in model['model'] 
                                             for model in recommendations['primary_models']):
        recommendations['preprocessing'].append('Feature scaling (StandardScaler for normal distribution, RobustScaler for outliers)')
    
    return recommendations

# DEVELOPER: Update these variables for your specific dataset
target_column = 'target'  # Replace with your actual target column name
problem_type = 'classification'  # Change to 'regression' for continuous targets

# Generate intelligent model recommendations
model_recommendations = recommend_models(dataset_analysis, target_column, problem_type)

# Display recommendations with reasoning
print("\n🤖 === MODEL RECOMMENDATIONS ===")
print("Primary Models (recommended):")
for i, model in enumerate(model_recommendations['primary_models'], 1):
    print(f"  {i}. {model['model']}: {model['reason']}")

if model_recommendations['alternative_models']:
    print("\nAlternative Models (consider if primary models underperform):")
    for i, model in enumerate(model_recommendations['alternative_models'], 1):
        print(f"  {i}. {model['model']}: {model['reason']}")

print("\n🔧 Recommended Preprocessing Steps:")
for i, step in enumerate(model_recommendations['preprocessing'], 1):
    print(f"  {i}. {step}")
```

### Step 4: Dataset Summary Report

**Purpose:** Generate comprehensive analysis summary with actionable next steps for development.

```python
# Step 4: Dataset Summary Report - Comprehensive overview and next steps
def generate_dataset_summary(df, target_column, dataset_analysis, model_recommendations):
    """
    Generate comprehensive dataset summary report
    """
    print("=" * 80)
    print("DATASET SUMMARY REPORT")
    print("=" * 80)
    
    # Basic info
    print(f"Dataset: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Target Variable: {target_column}")
    print(f"Problem Type: {model_recommendations.get('problem_type', 'TBD')}")
    
    # Target distribution
    print(f"\nTarget Distribution:")
    if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() <= 10:
        target_counts = df[target_column].value_counts()
        for value, count in target_counts.items():
            print(f"  {value}: {count} ({count/len(df):.1%})")
        
        # Check for class imbalance
        if len(target_counts) == 2:
            minority_ratio = target_counts.min() / target_counts.sum()
            if minority_ratio < 0.3:
                print(f"  ⚠️ Class imbalance detected (minority class: {minority_ratio:.1%})")
    else:
        print(f"  Mean: {df[target_column].mean():.3f}")
        print(f"  Std: {df[target_column].std():.3f}")
        print(f"  Range: {df[target_column].min():.3f} to {df[target_column].max():.3f}")
    
    # Data quality
    print(f"\nData Quality:")
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    print(f"  Missing Values: {missing_pct:.1f}% overall")
    print(f"  Duplicate Rows: {df.duplicated().sum()}")
    
    # Feature overview
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)
    
    print(f"\nFeature Overview:")
    print(f"  Numeric Features: {len(numeric_features)}")
    print(f"  Categorical Features: {len(categorical_features)}")
    
    # Model recommendations
    print(f"\nRecommended Models:")
    for i, model in enumerate(model_recommendations['primary_models'], 1):
        print(f"  {i}. {model['model']}")
    
    print(f"\nNext Steps:")
    print(f"  1. Handle missing values and outliers")
    print(f"  2. Encode categorical variables")
    print(f"  3. Split data and scale features")
    print(f"  4. Train and evaluate recommended models")
    print(f"  5. Perform hyperparameter tuning")
    
    return {
        'target_column': target_column,
        'problem_type': model_recommendations.get('problem_type', 'TBD'),
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 1,
        'recommended_models': model_recommendations['primary_models']
    }

# Generate summary (replace target_column with actual target)
summary = generate_dataset_summary(df, target_column, dataset_analysis, model_recommendations)
```

---

## Phase 2: Training Pipeline (Steps 5-16)

This phase implements the actual ML training pipeline with proper preprocessing, model training, and evaluation.

### Phase 2 Step Summary:
5. **Data Preprocessing**: Create pipeline-based preprocessing to prevent data leakage
6. **Train-Test Split**: Properly split data with stratification for classification  
7. **Model Training**: Train multiple recommended algorithms with cross-validation
8. **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV
9. **Cross-Validation**: Validate model performance with robust CV strategy
10. **Model Comparison**: Compare different algorithms and select best performer
11. **Feature Importance**: Analyze and extract feature importance/coefficients
12. **Model Persistence**: Save best model, metadata, and preprocessing pipeline
13. **Training Metrics**: Calculate and log comprehensive training metrics
14. **Model Validation**: Validate final model on held-out data
15. **Performance Analysis**: Analyze model behavior and potential issues
16. **Training Report**: Generate training summary with model selection rationale

### Step 5: Data Preprocessing

**Purpose:** Create pipeline-based preprocessing to handle missing values, encode categories, and scale features while preventing data leakage.

```python
# Step 5: Data Preprocessing - Pipeline-based preprocessing to prevent data leakage
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_preprocessing_pipeline(X, y, problem_type):
    """
    Create comprehensive preprocessing pipeline
    
    CRITICAL: Using pipelines ensures preprocessing is applied consistently
    to training and test data, preventing data leakage that would
    inflate performance estimates.
    """
    # Identify column types for appropriate preprocessing
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"📊 Preprocessing Setup:")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    
    # Numeric preprocessing pipeline
    numeric_pipeline = Pipeline([
        # Impute missing values with median (robust to outliers)
        ('imputer', SimpleImputer(strategy='median')),
        # Scale features for algorithms sensitive to magnitude (LogReg, SVM, Neural Networks)
        ('scaler', StandardScaler())  # Use RobustScaler if many outliers detected
    ])
    
    # Categorical preprocessing pipeline  
    categorical_pipeline = Pipeline([
        # Fill missing categorical values with 'unknown' category
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        # One-hot encode categories (drop='first' prevents multicollinearity)
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine preprocessing pipelines using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Handle target variable encoding for classification problems
    if problem_type == 'classification' and y.dtype == 'object':
        print("🎯 Encoding target variable (string labels → numeric)")
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"  Classes: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        return preprocessor, y_encoded, label_encoder
    
    return preprocessor, y, None

# Separate features and target - NEVER include target in feature matrix
X = df.drop(target_column, axis=1)
y = df[target_column]

print(f"🔧 Creating preprocessing pipeline...")
print(f"Features: {X.shape[1]} columns")
print(f"Target: '{target_column}' with {y.nunique()} unique values")

# Create preprocessing pipeline
preprocessor, y_processed, label_encoder = create_preprocessing_pipeline(X, y, problem_type)
```

### Step 6: Train-Test Split

**Purpose:** Split data into training and test sets while maintaining target distribution for reliable performance estimation.

```python
# Step 6: Train-Test Split - Proper data splitting for unbiased evaluation
from sklearn.model_selection import train_test_split

print("📊 Splitting data into train/test sets...")

# Use stratification for classification to maintain class balance
if problem_type == 'classification':
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_processed, 
        test_size=0.2,          # 80% training, 20% testing
        random_state=42,        # Reproducible results
        stratify=y_processed    # Maintain class proportions in both sets
    )
    print("✅ Stratified split applied to maintain class balance")
else:
    # Regular split for regression problems
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_processed, 
        test_size=0.2, 
        random_state=42
    )
    print("✅ Random split applied for regression problem")

print(f"📈 Data split results:")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Verify target distribution is maintained (critical for classification)
if problem_type == 'classification':
    print(f"\n🎯 Target distribution verification:")
    train_dist = np.bincount(y_train) / len(y_train)
    test_dist = np.bincount(y_test) / len(y_test)
    print(f"  Training: {train_dist}")
    print(f"  Test:     {test_dist}")
    print(f"  Difference: {np.abs(train_dist - test_dist).max():.4f} (should be < 0.05)")
```

### Step 7-16: Model Training and Evaluation

**Purpose:** Train multiple algorithms, perform cross-validation, select best model, and save artifacts for production use.

```python
# Step 7-16: Model Training and Evaluation - Complete training pipeline with model selection
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import pickle
import os

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, 
                            recommended_models, problem_type, save_format='joblib'):
    """
    Train multiple models and select the best performer
    
    This function implements the complete training pipeline:
    1. Trains each recommended model with cross-validation
    2. Compares performance across models
    3. Selects best model based on test performance
    4. Saves model artifacts for production deployment
    """
    print("🚀 Starting model training pipeline...")
    results = {}
    
    # Train each recommended model
    for i, model_info in enumerate(recommended_models, 1):
        model_name = model_info['model']
        model_params = model_info['parameters']
        
        print(f"\n📊 Training model {i}/{len(recommended_models)}: {model_name}")
        
        # Dynamic model initialization based on recommendation
        if model_name == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**model_params)
        elif model_name == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**model_params)
        elif model_name == 'GradientBoostingClassifier':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(**model_params)
        elif model_name == 'LinearRegression':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(**model_params)
        elif model_name == 'Ridge':
            from sklearn.linear_model import Ridge
            model = Ridge(**model_params)
        elif model_name == 'RandomForestRegressor':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**model_params)
        # DEVELOPER: Add more models here as needed
        else:
            print(f"❌ Model {model_name} not implemented, skipping...")
            continue
        
        # Create complete pipeline (preprocessing + model)
        # This ensures consistent preprocessing in production
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier' if problem_type == 'classification' else 'regressor', model)
        ])
        
        # Perform cross-validation for robust performance estimation
        print("  🔄 Running cross-validation...")
        if problem_type == 'classification':
            cv_scores = cross_val_score(full_pipeline, X_train, y_train, 
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                                      scoring='accuracy')
            cv_metric = 'accuracy'
        else:
            cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='r2')
            cv_metric = 'r2'
        
        # Train on full training set for final model
        print("  🎯 Training on full dataset...")
        full_pipeline.fit(X_train, y_train)
        
        # Evaluate on held-out test set for unbiased performance estimate
        y_pred = full_pipeline.predict(X_test)
        
        if problem_type == 'classification':
            test_score = accuracy_score(y_test, y_pred)
            test_metric = 'accuracy'
        else:
            test_score = r2_score(y_test, y_pred)
            test_metric = 'r2'
        
        # Store comprehensive results for model comparison
        results[model_name] = {
            'pipeline': full_pipeline,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_score': test_score,
            'predictions': y_pred
        }
        
        # Display performance summary
        print(f"  ✅ CV {cv_metric}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  📈 Test {test_metric}: {test_score:.4f}")
    
    # Select best performing model based on test set performance
    print(f"\n🏆 Model Selection Results:")
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_score'])
    best_pipeline = results[best_model_name]['pipeline']
    
    print(f"  Best model: {best_model_name}")
    print(f"  Best test score: {results[best_model_name]['test_score']:.4f}")
    
    # Model comparison summary
    print(f"\n📊 All Models Performance:")
    for name, result in sorted(results.items(), key=lambda x: x[1]['test_score'], reverse=True):
        print(f"  {name}: {result['test_score']:.4f} (CV: {result['cv_mean']:.4f} ± {result['cv_std']:.4f})")
    
    # Save best model and metadata for production use
    print(f"\n💾 Saving model artifacts...")
    os.makedirs('models', exist_ok=True)
    
    # Save model using specified format (joblib recommended for sklearn models)
    if save_format == 'joblib':
        model_path = f'models/best_model.pkl'
        joblib.dump(best_pipeline, model_path)
    else:  # pickle format
        model_path = f'models/best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_pipeline, f)
    
    # Save comprehensive metadata for model deployment and monitoring
    metadata = {
        'model_name': best_model_name,
        'cv_score': results[best_model_name]['cv_mean'],
        'cv_std': results[best_model_name]['cv_std'], 
        'test_score': results[best_model_name]['test_score'],
        'feature_names': X_train.columns.tolist(),
        'label_encoder': label_encoder if label_encoder else None,
        'problem_type': problem_type,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'all_results': {name: {'cv_mean': res['cv_mean'], 'test_score': res['test_score']} 
                       for name, res in results.items()}
    }
    
    # Save metadata
    if save_format == 'joblib':
        joblib.dump(metadata, 'models/model_metadata.pkl')
    else:
        with open('models/model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Metadata saved to: models/model_metadata.pkl")
    
    return best_pipeline, results, best_model_name

# Execute complete training pipeline
print("🎯 Executing training pipeline with recommended models...")
best_model, all_results, best_model_name = train_and_evaluate_models(
    X_train, X_test, y_train, y_test, preprocessor, 
    model_recommendations['primary_models'], problem_type
)

print(f"\n🎉 Training completed successfully!")
print(f"Best model: {best_model_name}")
print(f"Ready for Phase 3: Testing and Evaluation")
```

---

## Phase 3: Testing and Evaluation (Steps 17-28)

This phase implements comprehensive model testing following standardized ML evaluation protocols.

### Phase 3 Step Summary:
17. **Load Trained Model**: Load saved model and metadata for evaluation
18. **Test Set Preparation**: Prepare test data and generate predictions  
19. **Basic Performance Metrics**: Calculate accuracy, precision, recall, F1-score
20. **Advanced Metrics**: ROC-AUC, precision-recall curves, clinical metrics
21. **Confusion Matrix Analysis**: Detailed classification breakdown and interpretation
22. **Statistical Significance**: Confidence intervals and statistical tests
23. **Error Analysis**: Analyze misclassified samples and failure modes
24. **Feature Importance Validation**: Verify feature importance on test set
25. **Threshold Analysis**: Optimize decision thresholds for specific objectives
26. **Cross-Validation Comparison**: Compare test performance with CV results
27. **Results Documentation**: Generate comprehensive test report
28. **Framework Compliance**: Output standardized JSON results format

Following the ML Evaluation Framework from `ml_evaluation_framework.md`:

### Step 17-20: Comprehensive Model Testing

```python
import json
from datetime import datetime
from sklearn.metrics import *

def run_comprehensive_testing(model_path, X_test, y_test, results_dir='tests'):
    """
    Comprehensive testing following ML Evaluation Framework
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model
    if model_path.endswith('.pkl'):
        try:
            model = joblib.load(model_path)
        except:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
    
    # Initialize results structure following framework
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "framework_version": "1.0",
            "model_id": f"{best_model_name}_v1"
        },
        "tests": {},
        "summary": {}
    }
    
    # Model loading test
    results["tests"]["model_loading"] = {
        "status": "PASS",
        "model_type": str(type(model).__name__),
        "feature_count": X_test.shape[1],
        "test_samples": len(X_test)
    }
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    if problem_type == 'classification':
        # Classification tests
        results["tests"]["accuracy"] = {
            "value": accuracy_score(y_test, y_pred),
            "status": "PASS"
        }
        
        results["tests"]["precision_recall_f1"] = {
            "precision_macro": precision_score(y_test, y_pred, average='macro'),
            "recall_macro": recall_score(y_test, y_pred, average='macro'),
            "f1_macro": f1_score(y_test, y_pred, average='macro'),
            "status": "PASS"
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results["tests"]["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "status": "PASS"
        }
        
        # ROC-AUC for binary classification
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            results["tests"]["roc_auc"] = {
                "value": roc_auc_score(y_test, y_proba),
                "status": "PASS"
            }
            
            # Clinical metrics
            tn, fp, fn, tp = cm.ravel()
            results["tests"]["clinical_metrics"] = {
                "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp),
                "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
                "status": "PASS"
            }
    
    else:  # regression
        results["tests"]["mse_rmse"] = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "status": "PASS"
        }
        
        results["tests"]["mae"] = {
            "value": mean_absolute_error(y_test, y_pred),
            "status": "PASS"
        }
        
        results["tests"]["r2_score"] = {
            "value": r2_score(y_test, y_pred),
            "status": "PASS"
        }
    
    # Calculate summary
    total_tests = len(results["tests"])
    passed_tests = sum(1 for test in results["tests"].values() 
                      if isinstance(test, dict) and test.get("status") == "PASS")
    
    results["summary"] = {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": total_tests - passed_tests,
        "overall_status": "PASS" if passed_tests == total_tests else "FAIL"
    }
    
    # Add performance grade
    if problem_type == 'classification':
        primary_metric = results["tests"].get("roc_auc", results["tests"]["accuracy"])["value"]
        if primary_metric >= 0.95:
            grade = "EXCELLENT"
        elif primary_metric >= 0.85:
            grade = "GOOD"
        elif primary_metric >= 0.75:
            grade = "FAIR"
        else:
            grade = "POOR"
    else:
        primary_metric = results["tests"]["r2_score"]["value"]
        if primary_metric >= 0.9:
            grade = "EXCELLENT"
        elif primary_metric >= 0.8:
            grade = "GOOD"
        elif primary_metric >= 0.7:
            grade = "FAIR"
        else:
            grade = "POOR"
    
    results["summary"]["key_metrics"] = {
        "primary_metric": "roc_auc" if problem_type == 'classification' else "r2_score",
        "primary_value": primary_metric,
        "performance_grade": grade
    }
    
    # Save results
    results_file = os.path.join(results_dir, 'model_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Testing completed. Results saved to {results_file}")
    print(f"Overall Status: {results['summary']['overall_status']}")
    print(f"Performance Grade: {grade}")
    
    return results

# Run comprehensive testing
test_results = run_comprehensive_testing('models/best_model.pkl', X_test, y_test)
```

### Step 21-28: Additional Analysis and Validation

```python
def perform_additional_analysis(model, X_test, y_test, results_dir='tests'):
    """
    Additional model analysis and validation
    """
    print("\n=== ADDITIONAL MODEL ANALYSIS ===")
    
    # Feature importance (if available)
    if hasattr(model.named_steps.get('classifier', model.named_steps.get('regressor')), 'feature_importances_'):
        feature_importance = model.named_steps.get('classifier', model.named_steps.get('regressor')).feature_importances_
        
        # Get feature names after preprocessing
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importances:")
        print(importance_df.head(10))
        
        # Save feature importance
        importance_df.to_csv(os.path.join(results_dir, 'feature_importance.csv'), index=False)
    
    # Error analysis
    y_pred = model.predict(X_test)
    errors = np.abs(y_test - y_pred) if problem_type == 'regression' else (y_test != y_pred)
    
    print(f"\nError Analysis:")
    if problem_type == 'regression':
        print(f"  Mean Absolute Error: {np.mean(errors):.4f}")
        print(f"  Max Error: {np.max(errors):.4f}")
        print(f"  Error Std: {np.std(errors):.4f}")
    else:
        print(f"  Misclassified samples: {np.sum(errors)} out of {len(y_test)} ({np.mean(errors):.1%})")
    
    # Prediction confidence analysis (for classification with probabilities)
    if problem_type == 'classification' and hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        confidence = np.max(y_proba, axis=1)
        
        print(f"\nPrediction Confidence:")
        print(f"  Mean confidence: {np.mean(confidence):.3f}")
        print(f"  Min confidence: {np.min(confidence):.3f}")
        print(f"  Low confidence predictions (<0.6): {np.sum(confidence < 0.6)}")

# Perform additional analysis
perform_additional_analysis(best_model, X_test, y_test)
```

---

## Phase 4: Interactive TUI Application (Steps 29-35)

This phase creates a professional interactive application for real-world model deployment and usage.

### Phase 4 Step Summary:
29. **TUI Framework Setup**: Initialize rich console interface and model loading
30. **Welcome Interface**: Create professional welcome screen with model performance
31. **Input Data Collection**: Interactive feature input with validation and descriptions
32. **Prediction Engine**: Real-time prediction with probability estimates
33. **Results Display**: Professional output with risk levels and recommendations
34. **Feature Contribution Analysis**: Show which features drive predictions
35. **Application Flow Management**: Handle errors, confirmations, and user experience

### Step 29-30: TUI Application Framework

```python
# TUI Application using rich library
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, FloatPrompt, IntPrompt
from rich import print as rprint

class MLPredictor:
    def __init__(self, model_path, metadata_path):
        self.console = Console()
        self.model = None
        self.metadata = None
        self.load_model(model_path, metadata_path)
    
    def load_model(self, model_path, metadata_path):
        """Load trained model and metadata"""
        try:
            # Load model
            if model_path.endswith('.pkl'):
                try:
                    self.model = joblib.load(model_path)
                except:
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
            
            # Load metadata
            try:
                self.metadata = joblib.load(metadata_path)
            except:
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            
            return True
        except FileNotFoundError:
            self.console.print("[red]Error: Model files not found![/red]")
            return False
    
    def display_welcome(self):
        """Display welcome screen"""
        problem_type = self.metadata.get('problem_type', 'Unknown')
        model_name = self.metadata.get('model_name', 'Unknown')
        test_score = self.metadata.get('test_score', 0)
        
        welcome_panel = Panel.fit(
            f"[bold blue]ML Prediction System[/bold blue]\\n"
            f"[dim]AI-Powered {problem_type.title()} Tool[/dim]\\n\\n"
            f"Model: {model_name}\\n"
            f"Test Score: {test_score:.3f}",
            title="🤖 Welcome",
            border_style="blue"
        )
        self.console.print(welcome_panel)
    
    def collect_input_data(self):
        """Collect input data for prediction"""
        self.console.print("\\n[bold green]Data Collection[/bold green]")
        
        feature_names = self.metadata.get('feature_names', [])
        input_data = {}
        
        for feature in feature_names:
            # Simple input collection - enhance based on feature types
            value = FloatPrompt.ask(f"Enter {feature}")
            input_data[feature] = value
        
        return input_data
    
    def make_prediction(self, input_data):
        """Make prediction using the loaded model"""
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        
        # Get probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(input_df)[0]
            probability = np.max(proba)
        
        return prediction, probability
    
    def display_results(self, prediction, probability=None):
        """Display prediction results"""
        problem_type = self.metadata.get('problem_type', 'Unknown')
        
        if problem_type == 'classification':
            # Handle label encoding
            label_encoder = self.metadata.get('label_encoder')
            if label_encoder:
                prediction = label_encoder.inverse_transform([prediction])[0]
            
            result_text = f"Prediction: [bold]{prediction}[/bold]"
            if probability:
                result_text += f"\\nConfidence: [bold]{probability:.1%}[/bold]"
        else:
            result_text = f"Predicted Value: [bold]{prediction:.3f}[/bold]"
        
        result_panel = Panel(
            result_text,
            title="🎯 Prediction Result",
            border_style="green"
        )
        self.console.print(result_panel)
    
    def run_application(self):
        """Main application flow"""
        self.console.clear()
        self.display_welcome()
        
        while True:
            try:
                # Collect input data
                input_data = self.collect_input_data()
                
                # Make prediction
                prediction, probability = self.make_prediction(input_data)
                
                # Display results
                self.display_results(prediction, probability)
                
                # Ask for another prediction
                if not Confirm.ask("\\nWould you like to make another prediction?"):
                    break
                    
            except KeyboardInterrupt:
                self.console.print("\\n[yellow]Application interrupted by user.[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")
                if not Confirm.ask("Would you like to try again?"):
                    break
        
        self.console.print("\\n[blue]Thank you for using the ML Prediction System![/blue]")

# Usage
if __name__ == "__main__":
    app = MLPredictor('models/best_model.pkl', 'models/model_metadata.pkl')
    
    if app.model is None:
        print("Failed to load model files.")
        exit(1)
    
    app.run_application()
```

### Step 31-35: Enhanced TUI Features

```python
def create_enhanced_tui():
    """
    Enhanced TUI with additional features
    """
    
    # Add batch prediction capability
    def batch_predict_from_csv(self, csv_path):
        """Predict from CSV file"""
        try:
            data = pd.read_csv(csv_path)
            predictions = self.model.predict(data)
            
            # Save predictions
            data['predictions'] = predictions
            output_path = csv_path.replace('.csv', '_predictions.csv')
            data.to_csv(output_path, index=False)
            
            self.console.print(f"[green]Batch predictions saved to {output_path}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error in batch prediction: {e}[/red]")
    
    # Add model performance display
    def display_model_performance(self):
        """Display model performance metrics"""
        if hasattr(self, 'test_results'):
            performance_table = Table(title="Model Performance")
            performance_table.add_column("Metric", style="bold")
            performance_table.add_column("Value", justify="right")
            
            for test_name, test_data in self.test_results['tests'].items():
                if isinstance(test_data, dict) and 'value' in test_data:
                    performance_table.add_row(test_name, f"{test_data['value']:.4f}")
            
            self.console.print(performance_table)
    
    # Add feature importance display
    def display_feature_importance(self):
        """Display feature importance if available"""
        if hasattr(self.model.named_steps.get('classifier', self.model.named_steps.get('regressor')), 'feature_importances_'):
            # Implementation for feature importance display
            pass

print("Enhanced TUI template created successfully!")
```

---

## Usage Instructions

### Quick Start

1. **Replace placeholders:**
   - Update `your_dataset.csv` with your dataset path
   - Set `target_column` to your target variable name
   - Set `problem_type` to 'classification' or 'regression'

2. **Run the analysis:**
   ```python
   python ml_pipeline.py
   ```

3. **Use the TUI application:**
   ```python
   python tui_app.py
   ```

### Customization Options

- **Model Selection:** Add/remove models in the recommendation engine
- **Preprocessing:** Modify preprocessing steps based on data characteristics
- **Evaluation Metrics:** Customize metrics in the testing phase
- **TUI Features:** Enhance the interface with additional functionality

### Framework Integration

This template integrates with the ML Evaluation Framework (`ml_evaluation_framework.md`) for:
- Standardized testing protocols
- Consistent result formats
- Performance grading
- Multi-model comparison capabilities

---

## File Structure

```
project/
├── ml_pipeline.py          # Main pipeline script
├── tui_app.py             # Interactive TUI application
├── your_dataset.csv       # Your dataset
├── models/
│   ├── best_model.pkl     # Trained model
│   └── model_metadata.pkl # Model metadata
├── tests/
│   ├── model_test_results.json  # Framework-compliant results
│   └── feature_importance.csv   # Feature analysis
└── ml_evaluation_framework.md   # Testing framework reference
```

This template provides a complete foundation for supervised ML projects while maintaining flexibility for customization and integration with existing workflows.