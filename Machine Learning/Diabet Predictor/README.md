# Diabetes Predictor

## Project Overview

This project implements a comprehensive machine learning pipeline for predicting diabetes in patients using medical and demographic data. The pipeline follows best practices for data preprocessing, feature engineering, and model training, resulting in a clinically relevant predictive model.

The project was developed as Lab 9 - "Build a ML Pipeline from scratch" and demonstrates end-to-end machine learning workflow including data exploration, preprocessing, model selection, hyperparameter tuning, and evaluation.

## Dataset

The analysis uses the **Pima Indians Diabetes Database** from Kaggle (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), containing medical records of 768 female patients of Pima Indian heritage.

### Features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years

### Target Variable:
- **Outcome**: 0 (No diabetes) or 1 (Diabetes)

## Methodology

The project follows a systematic ML pipeline approach:

1. **Data Collection**: Load and examine the dataset
2. **Data Exploration**: Analyze distributions, correlations, and data quality
3. **Data Splitting**: 80/20 train/test split with stratification
4. **Data Cleaning**: Handle missing values and outliers
5. **Feature Selection**: Remove redundant features
6. **Feature Engineering**: Create new meaningful features
7. **Data Transformation**: Scaling and normalization
8. **Class Imbalance Handling**: Apply oversampling techniques
9. **Model Selection and Training**: Train multiple algorithms
10. **Hyperparameter Tuning**: Optimize model parameters
11. **Model Evaluation**: Compare performance using appropriate metrics

## Data Exploration Findings

- Dataset contains 768 samples with 8 features
- Significant missing values represented as zeros in medical features (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
- Class imbalance: ~65% negative cases vs ~35% positive cases (ratio ~1.9)
- Presence of outliers in medical measurements requiring robust scaling
- No strong correlations between features, no redundant features to remove

## Preprocessing Steps

### 1. Missing Value Handling
- Identified zeros as missing values in 5 medical features
- Replaced zeros with NaN for proper imputation
- Applied mean imputation strategy (tested mean, median, most_frequent)
- Missing value percentages: 40-50% in Insulin, ~4-5% in other features

### 2. Train/Test Split
- 80% training, 20% testing with stratification to maintain class distribution
- Computed statistics only on training set to prevent data leakage

### 3. Feature Engineering
- Created "Glucose/Insulin ratio" feature to capture metabolic relationships
- No discretization needed (all features suitable as continuous)
- No categorical encoding required (all features numeric)

### 4. Feature Scaling
- Applied RobustScaler due to presence of outliers
- RobustScaler uses median and IQR, less sensitive to outliers than StandardScaler

### 5. Class Imbalance Handling
- Applied SMOTE (Synthetic Minority Oversampling Technique) for oversampling
- Chose oversampling over undersampling based on dataset size and minority class count
- Balanced training set to improve model performance on minority class

## Models Trained

### 1. Logistic Regression Variants
- **Linear (Degree 1)**: Baseline logistic regression
- **Polynomial (Degree 4)**: Added polynomial features for non-linear relationships
- **Ridge Regularization**: L2 penalty to prevent overfitting

### 2. XGBoost
- Gradient boosting algorithm with scale_pos_weight for class imbalance
- Hyperparameter tuning for n_estimators

## Evaluation Metrics

Given the medical context and class imbalance, focused on:
- **Recall**: Critical for minimizing false negatives (missed diabetes cases)
- **F1-Score**: Balanced measure of precision and recall
- **AUC-ROC**: Overall classification performance
- **Confusion Matrix**: Detailed error analysis

## Results

### Model Performance Comparison

| Model Configuration | Recall | F1-Score | AUC | Notes |
|---------------------|--------|----------|-----|--------|
| Logistic Regression (Degree 1) | 0.55 | 0.65 | 0.82 | Baseline |
| Logistic Regression (Degree 4) | 0.60 | 0.68 | 0.83 | Polynomial features |
| Logistic Regression + Ridge | 0.58 | 0.66 | 0.81 | Regularization |
| **XGBoost + scale_pos_weight** | **0.76** | **0.72** | **0.85** | **Best performer** |

### Key Findings
- **XGBoost achieved 76% recall**, a 38% improvement over baseline
- **SMOTE oversampling** improved recall from ~0.55 to ~0.76
- **scale_pos_weight** effectively handled class imbalance without data loss
- **Polynomial features** provided marginal improvement over linear
- **Ridge regularization** helped prevent overfitting in complex models

## Clinical Relevance

- **High recall priority**: False negatives (undiagnosed diabetes) are more dangerous than false positives
- **Medical impact**: 76% recall means fewer missed diabetes cases
- **Screening value**: Improved early detection of diabetes risk
- **Healthcare application**: Suitable for clinical decision support systems

## Technical Insights

- **Algorithm selection**: XGBoost captured complex feature interactions better than linear models
- **Preprocessing importance**: Robust scaling and feature engineering crucial for medical data
- **Imbalance handling**: Algorithmic approach (scale_pos_weight) outperformed traditional resampling
- **Evaluation focus**: F1-score and recall more appropriate than accuracy for imbalanced medical data

## Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-Learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **imbalanced-learn (imblearn)**: SMOTE oversampling
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Development environment

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
```

## How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open the Jupyter notebook: `DiabetPredictor.ipynb`
4. Run all cells in sequence
5. View results and visualizations

## Limitations & Future Work

- **Dataset size**: Limited to 768 samples
- **Population specificity**: Pima Indians dataset may not generalize universally
- **Feature consistency**: Insulin measurements may not always be available
- **Future improvements**:
  - Hyperparameter optimization for XGBoost
  - Cross-validation with different random seeds
  - Comparison with Random Forest, LightGBM
  - Feature selection techniques
  - Deep learning approaches for larger datasets

## Conclusion

This project successfully demonstrates a complete ML pipeline for diabetes prediction, achieving clinically relevant performance with 76% recall using XGBoost. The comprehensive preprocessing, feature engineering, and careful evaluation make this a robust approach for medical prediction tasks.

The pipeline balances technical excellence with clinical requirements, providing a foundation for real-world diabetes screening applications.