# California Housing Price Prediction

A comprehensive machine learning project for predicting median house values in California using the famous California Housing dataset.

## 📊 Project Overview

This project implements a complete machine learning pipeline to predict median house prices across different districts in California. The model uses various regression techniques and evaluation metrics to achieve optimal performance.

## 🎯 Objective

Predict the median house value (`MedHouseVal`) for California housing districts based on various socio-economic and geographic features.

## 📋 Dataset Description

**Source**: California Housing Dataset (scikit-learn)
- **Samples**: 20,640 housing districts
- **Features**: 8 numerical features
- **Target**: Median house value in hundreds of thousands of dollars

### Features:
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group  
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

## 🔧 Methodology

### 1. Data Preprocessing
- **Missing Values**: No missing values detected
- **Outlier Treatment**: Applied RobustScaler to handle outliers in features like `MedInc`, `HouseAge`, `AveOccup`, and `Population`
- **Scaling**: RobustScaler used for normalization

### 2. Model Training & Evaluation
Trained and compared multiple regression models:

| Model | Cross-Validation R² | Test R² | RMSE | MAE |
|-------|-------------------|---------|------|-----|
| Linear Regression | ~0.60 | - | 0.745 | 0.533 |
| Ridge Regression | ~0.60 | - | 0.745 | 0.533 |
| Lasso Regression | ~0.60 | - | 0.745 | 0.532 |
| Random Forest | ~0.80 | - | - | - |
| **XGBoost** | **~0.84** | **0.844** | **0.204** | **0.291** |

### 3. Model Selection
- **Primary Selection Criteria**: Cross-validation scores (10-fold CV)
- **Best Model**: XGBoost Regressor based on highest CV R² score
- **Hyperparameter Optimization**: RandomizedSearchCV for XGBoost parameter tuning

### 4. Hyperparameter Tuning
Used RandomizedSearchCV for XGBoost optimization:
```python
param_dist = {
    'n_estimators': [100, 200, 400, 600],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}
```

## 📈 Results

### Final Model Performance (XGBoost)
- **Cross-Validation R²**: ~0.84
- **Test Set R²**: 0.844
- **RMSE**: 0.204
- **MAE**: 0.291

### Performance Interpretation
- The model explains **84.4%** of the variance in house prices
- Performance is **excellent** for real estate prediction (benchmark: >0.8 = excellent)
- Only **15.6%** of variance remains unexplained - this is a very strong result!

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation
  - `numpy` - Numerical operations
  - `scikit-learn` - Machine learning algorithms
  - `xgboost` - Gradient boosting
  - `seaborn` & `matplotlib` - Data visualization

## 📁 File Structure

```
California Housing/
├── CaliforniaHousing.ipynb    # Main analysis notebook
├── README.md                   # This file
└── [model files if saved]
```

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn xgboost seaborn matplotlib
   ```

2. **Run the Notebook**:
   ```bash
   jupyter notebook CaliforniaHousing.ipynb
   ```

3. **Execute Cells**: Run all cells sequentially to reproduce the analysis

## 📊 Key Insights

1. **XGBoost outperformed** linear models significantly
2. **Cross-validation** was crucial for reliable model selection
3. **Feature scaling** with RobustScaler improved model performance
4. **Geographic features** (Lat/Long) were removed, but could be valuable for future improvements with proper geographic clustering

## 🔮 Future Improvements

1. **Feature Engineering**:
   - Create polynomial features
   - Geographic clustering
   - Domain-specific ratios (rooms per person, etc.)

2. **Advanced Models**:
   - Ensemble methods (stacking)
   - Neural networks
   - CatBoost/LightGBM

3. **Feature Recovery**:
   - Reintroduce geographic features with proper engineering
   - Create location-based clusters

## 👥 Author

**Dorotea Monaco**
- Machine Learning Project
- Politecnico di Torino

*Last Updated: October 2025*