# Hospital Length of Stay (LOS) Prediction - Step-by-Step Technical Guide & Explanation

This document provides a comprehensive technical explanation for each step (Tasks 12–29) implemented in the Jupyter Notebook [`01_los_prediction.ipynb`](file:///d:/mitacs_portfolio_2027/Classical%20ML%20Time-Series/01_los_prediction.ipynb).

---

## 📌 Table of Contents
1. [Task 12: Missing Data Analysis](#task-12-missing-data-analysis)
2. [Task 13: Numerical Imputation with Median](#task-13-numerical-imputation-with-median)
3. [Task 14: Categorical Imputation with Mode](#task-14-categorical-imputation-with-mode)
4. [Task 15: Categorical Encoding (Label Encoding)](#task-15-categorical-encoding-label-encoding)
5. [Task 16: Feature Selection & Dropping Unnecessary Columns](#task-16-feature-selection--dropping-unnecessary-columns)
6. [Task 17: Defining Feature Matrix (X) and Target Vector (y)](#task-17-defining-feature-matrix-x-and-target-vector-y)
7. [Task 18: Train-Test Dataset Splitting (80/20)](#task-18-train-test-dataset-splitting-8020)
8. [Task 19: Saving Cleaned Datasets](#task-19-saving-cleaned-datasets)
9. [Tasks 20–25: Linear Regression Model & Metrics](#tasks-2025-linear-regression-model--metrics)
10. [Tasks 26–29: Decision Tree Regressor & Metrics](#tasks-2629-decision-tree-regressor--metrics)
11. [Model Performance Comparison](#model-performance-comparison)

---

## Task 12: Missing Data Analysis

### Explanation
In healthcare machine learning pipelines, missing values can arise due to unrecorded patient procedures, optional clinical forms, or system logging delays. Before building predictive models, we must inspect missing value distributions across all variables.

### Implementation
```python
# Task 12: Find missing columns
print(df.isnull().sum())
print("\nTotal missing values:", df.isnull().sum().sum())
```

### Empirical Results
- **`InsurancePlanName`**: 72 missing entries
- **`Discharge Time`**: 238 missing entries
- **`Discharge Before 12PM`**: 238 missing entries
- **Total Missing Values**: **548 missing fields** across 500 patient records.

---

## Task 13: Numerical Imputation with Median

### Explanation
When handling missing values in numerical features, replacing `NaN` values with the **Median** is preferred over the Mean when the data distribution contains skewness or extreme outliers (such as hospital charges or extended stays). Median represents the 50th percentile and is robust against skewed distributions.

### Implementation
```python
# Task 13: Fill numerical missing with Median
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {col} with median: {median_val}")
```

---

## Task 14: Categorical Imputation with Mode

### Explanation
Categorical variables (e.g., insurance plan type, discharge timing indicators) cannot be imputed using mathematical averages. Instead, we use the **Mode**—the most frequently occurring category in that column—to fill missing values without distorting categorical frequency distributions.

### Implementation
```python
# Task 14: Fill categorical missing with Mode
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Filled {col} with mode: {mode_val}")
```

---

## Task 15: Categorical Encoding (Label Encoding)

### Explanation
Machine learning algorithms (such as Linear Regression and Decision Trees) require numerical matrix inputs. **Label Encoding** converts each distinct string category into an integer code ($0, 1, 2, \dots, N-1$).

### Columns Encoded
`Month`, `Nationality`, `Gender`, `DoctorLicense`, `DoctorName`, `Doctor Type`, `Doctor Status`, `Specialty`, `Insurance/Payer`, `InsurancePlanName`, `Payer Mix`, `Case type`, `Surgical Mix`, `Discharge Time`, `Discharge Before 12PM`.

### Implementation
```python
# Task 15: Encode Categorical Columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"Encoded column: {col}")
```

---

## Task 16: Feature Selection & Dropping Unnecessary Columns

### Explanation
Unique identifier columns (such as `Case_No`, `DoctorLicense`, `DoctorName`) exhibit near-unique cardinalities and carry no generalizable predictive signal for patient length of stay. Keeping them can lead to severe model overfitting. We drop these non-predictive identifiers from the feature set.

### Implementation
```python
# Task 16: Drop unnecessary columns
columns_to_drop = ['Case_No', 'DoctorLicense', 'DoctorName']

for col in columns_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"Dropped column: {col}")
```

---

## Task 17: Defining Feature Matrix (X) and Target Vector (y)

### Explanation
We separate our processed dataset into:
- **Feature Matrix ($X$)**: All 17 predictive clinical and administrative variables.
- **Target Vector ($y$)**: The continuous variable **`LOS`** (Length of Stay in days).

### Dimensions
- **$X$ shape**: `(500, 17)`
- **$y$ shape**: `(500,)`

### Implementation
```python
# Task 17: Define X and y
target_column = 'LOS'

X = df.drop(columns=[target_column])
y = df[target_column]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
```

---

## Task 18: Train-Test Dataset Splitting (80/20)

### Explanation
To evaluate how well our models generalize to unseen patient data, we perform an **80/20 train-test split**. Setting `random_state=42` ensures reproducible data partitioning.
- **Training Set (80%)**: 400 patient records used for model parameter learning.
- **Test Set (20%)**: 100 patient records reserved strictly for final model evaluation.

### Implementation
```python
# Task 18: Train-Test Split (80% Train, 20% Test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

---

## Task 19: Saving Cleaned Datasets

### Explanation
Preserving preprocessed training and testing splits in CSV format (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`) ensures pipeline modularity, reproducible benchmarking, and seamless integration with downstream model training scripts.

### Implementation
```python
# Task 19: Save cleaned train/test data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("Cleaned train/test data saved successfully!")
```

---

## Tasks 20–25: Linear Regression Model & Metrics

### Concept & Mathematical Foundation
**Linear Regression** models the target variable $y$ as a linear combination of input features:
$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$

### Evaluation Metrics Formulas
1. **Mean Absolute Error (MAE)**: Measures average absolute magnitude of prediction errors in days.
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
2. **Root Mean Square Error (RMSE)**: Penalizes larger prediction errors more heavily.
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### Implementation
```python
# Tasks 20-25: Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Task 21 & 22: Import & Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Task 23: Predict on test data
y_pred_lr = lr_model.predict(X_test)

# Task 24: Calculate MAE
mae_lr = mean_absolute_error(y_test, y_pred_lr)

# Task 25: Calculate RMSE
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("Linear Regression Trained Successfully!")
print(f"Linear Regression MAE: {mae_lr:.4f}")
print(f"Linear Regression RMSE: {rmse_lr:.4f}")
```

### Empirical Results
- **MAE**: **`0.7302`** days
- **RMSE**: **`0.9654`** days

---

## Tasks 26–29: Decision Tree Regressor & Metrics

### Concept & Mathematical Foundation
A **Decision Tree Regressor** recursively partitions the feature space into orthogonal regions by finding split points that minimize Variance / Mean Squared Error (MSE) within leaf nodes. Unlike linear regression, decision trees naturally capture non-linear feature interactions and non-monotonic relationships.

### Implementation
```python
# Tasks 26-29: Decision Tree Regressor Model
from sklearn.tree import DecisionTreeRegressor

# Task 26 & 27: Import & Train DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Task 28: Predict on test data
y_pred_dt = dt_model.predict(X_test)

# Task 29: Calculate MAE and RMSE for Decision Tree
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))

print("Decision Tree Regressor Trained Successfully!")
print(f"Decision Tree MAE: {mae_dt:.4f}")
print(f"Decision Tree RMSE: {rmse_dt:.4f}")
```

### Empirical Results
- **MAE**: **`0.4000`** days
- **RMSE**: **`0.9798`** days

---

## Model Performance Comparison

| Evaluation Metric | Linear Regression | Decision Tree Regressor | Winning Model |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | `0.7302` days | **`0.4000` days** | 🏆 **Decision Tree** |
| **Root Mean Square Error (RMSE)** | **`0.9654` days** | `0.9798` days | 🏆 **Linear Regression** |

### Key Analytical Takeaways
1. **Decision Tree Regressor** achieves a significantly lower **MAE (0.4000 days)** compared to Linear Regression (0.7302 days), demonstrating its capability to capture non-linear clinical pathways and categorical threshold effects in hospital length of stay data.
2. **Linear Regression** produces a slightly lower **RMSE (0.9654 days)**, indicating more stable predictions across extreme tail cases due to its global linear constraint.
