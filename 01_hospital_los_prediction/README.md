# Hospital Length of Stay (LOS) Prediction

## Introduction
Accurately predicting patient Length of Stay (LOS) is vital for healthcare resource management, bed allocation, and operational efficiency. This project develops and evaluates four regression models (Linear Regression, Decision Tree, Random Forest, and XGBoost) to forecast hospital stay duration based on patient demographics, clinical characteristics, and administrative data.

## Dataset and Preprocessing
The dataset comprises clinical and administrative patient records containing missing values, categorical features, and numerical variables.
- **Missing Value Imputation**: Numerical missing values were imputed using column medians, and categorical missing values were imputed using column modes.
- **Categorical Encoding**: LabelEncoding was applied to transform categorical variables (Specialty, Insurance Plan, Case Type, etc.) into numeric format.
- **Feature Selection & Split**: Identifier columns (`Case_No`, `DoctorLicense`, `DoctorName`) were dropped. Data was partitioned into 80% training (`X_train`) and 20% test (`X_test`) sets.

## Results
Model performance was evaluated on the unseen test set using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE):

| Model | MAE | RMSE |
| :--- | :---: | :---: |
| **Linear Regression** | 0.7302 | 0.9654 |
| **Decision Tree** | 0.4000 | 0.9798 |
| **Random Forest** | 0.2436 | 0.5691 |
| **XGBoost** | **0.2856** | **0.7009** |

**Conclusion:** XGBoost achieved the best overall predictive accuracy with the lowest RMSE (0.7009).

## How to Run
1. Open Jupyter Notebook in this directory:
   ```bash
   jupyter notebook 01_los_prediction.ipynb
   ```
2. Execute all cells sequentially or use Kernel -> Restart & Run All.
