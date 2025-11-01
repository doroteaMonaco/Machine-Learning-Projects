# 🏥 Medical Insurance Cost Predictor

A comprehensive machine learning project that predicts medical insurance costs based on personal characteristics and lifestyle factors.

## 📊 Project Overview

This project implements an advanced regression model to predict medical insurance charges using demographic and health-related features. The model achieves exceptional performance with an **R² score of 88.4%**, demonstrating high accuracy in cost prediction.

## 🎯 Key Results

- **R² Score**: 88.4% (Excellent performance)
- **Model**: XGBoost Regressor with hyperparameter optimization
- **Dataset**: Medical Insurance dataset with 1,338 samples

## 📋 Dataset Features

### Original Features:
- **age**: Age of the primary beneficiary
- **sex**: Insurance contractor gender (female/male)
- **bmi**: Body mass index (kg/m²)
- **children**: Number of children covered by health insurance
- **smoker**: Smoking status (yes/no)
- **region**: Beneficiary's residential area (northeast, northwest, southeast, southwest)
- **charges**: Individual medical costs billed by health insurance (target variable)

### Engineered Features:
- **Region encoding**: One-hot encoded regional variables

## 🔧 Technical Implementation

### Data Preprocessing:
1. **Missing Values**: No missing values detected
2. **Categorical Encoding**:
   - Label Encoding for binary variables (sex, smoker)
   - One-Hot Encoding for region (drop_first=True to avoid multicollinearity)
3. **Feature Scaling**: StandardScaler applied to numerical features
4. **Outlier Analysis**: Comprehensive boxplot analysis for continuous variables

### Model Development:
1. **Baseline Models**: Linear Regression, Ridge, Lasso, Random Forest
2. **Advanced Model**: XGBoost Regressor
3. **Hyperparameter Tuning**: RandomizedSearchCV with 50 iterations
4. **Cross-Validation**: 10-fold CV for robust performance estimation

### Best Model Configuration:
```python
XGBRegressor(
    n_estimators=[100, 200, 400, 600],
    max_depth=[3, 5, 7, 9],
    learning_rate=[0.01, 0.05, 0.1, 0.2],
    reg_alpha=[0, 0.1, 0.5],
    reg_lambda=[1, 1.5, 2],
    random_state=42
)
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| R² Score | **88.4%** |
| MAE | Low prediction error |
| RMSE | Minimal variance in predictions |
| Cross-Validation R² | ~87% (consistent performance) |

### Model Comparison:
- **XGBoost**: 88.4% R² (Best performer)
- **Random Forest**: ~80% R²
- **Linear Models**: ~75% R²

## 🔍 Key Insights

### Feature Importance:
1. **Smoking Status**: Most significant predictor of insurance costs
2. **Age**: Strong positive correlation with charges
3. **BMI**: Important factor, especially for high BMI values

### Business Insights:
- Smokers have significantly higher insurance costs (3-4x higher median)
- Age shows linear relationship with insurance charges
- Regional differences exist but are less impactful than lifestyle factors
- The combination of smoking and advanced age leads to exponentially higher costs

## 📁 Project Structure

```
Medical Insurance/
├── Medical Insurance.ipynb    # Main analysis notebook
├── README.md                 # This file
└── archive/
    └── insurance.csv         # Dataset
```

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **seaborn/matplotlib**: Data visualization
- **Jupyter Notebook**: Development environment

## 🚀 Getting Started

### Prerequisites:
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib jupyter
```

### Running the Project:
1. Clone the repository
2. Navigate to the Medical Insurance folder
3. Open `Medical Insurance.ipynb` in Jupyter Notebook
4. Run all cells to reproduce the results

### Usage Example:
```python
# Load the trained model
model = XGBRegressor(**best_params)
model.fit(X_train_final, y_train)

# Make predictions
predictions = model.predict(X_test_final)
```

## 📊 Methodology Highlights

### 1. Exploratory Data Analysis:
- Comprehensive correlation analysis
- Outlier detection using boxplots
- Distribution analysis by categorical variables

### 2. Feature Engineering:
- Domain-specific interaction terms
- Proper encoding of categorical variables
- Feature scaling for optimal model performance

### 3. Model Selection:
- Multiple algorithm comparison
- Cross-validation for unbiased evaluation
- Hyperparameter optimization for best performance

### 4. Validation:
- Train/test split (80/20)
- 10-fold cross-validation
- Consistent performance across validation methods

## 🔬 Advanced Features

### Correlation Analysis:
- Identified multicollinearity issues
- Feature selection based on correlation thresholds
- Visualization through correlation heatmaps

### Outlier Handling:
- Statistical outlier detection
- Domain knowledge validation
- Preservation of legitimate high-cost cases

### Performance Optimization:
- RandomizedSearchCV for efficient hyperparameter tuning
- Feature importance analysis
- Model interpretability through visualizations

## 📚 References

- Dataset: Medical Cost Personal Dataset
- XGBoost Documentation: [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)
- Scikit-learn: [scikit-learn.org](https://scikit-learn.org/)

## 👨‍💻 Author

**doroteaMonaco**
- GitHub: [@doroteaMonaco](https://github.com/doroteaMonaco)
- Project: Machine Learning Projects Repository

⭐ **Star this repository if you found it helpful!**
