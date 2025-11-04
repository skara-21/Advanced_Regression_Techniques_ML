# 🏠 House Price Prediction

The goal of this project is to predict residential house prices using various regression-based models. The project utilizes Sklearn, XGBoost, Feature Engineering, and MLflow for model training, while Dagshub is used as a platform for model versioning and logging.

## Project Structure

- `model_experiment.ipynb`  
  - Main file.
  - Contains the following sections:
    - **Data Cleaning**
    - **Feature Selection**
    - **Feature Engineering**
    - **Training**
    - **ML Flow Tracking**
    - **Result saving**

- `model_inference.ipynb`  
  - At the end of the previous file, I saved both the model and processed data.
  - In this file, I loaded the model from **Model Registry**
  - Imported the processed data and made predictions.
  - Formatted the final predictions for competition submission and uploaded them.
  - **Result: 0.13371**

- `README.md`  
  - Contains a brief overview of the project

## Technologies Used

- pandas, numpy
- scikit-learn
- XGBoost
- MLflow
- Dagshub

## Models

The following regression models were compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- XGBoost

Evaluation was performed using `R2 Score` and `RMSLE` with Cross-Validation (KFold).

## Model Evaluation

The best results were achieved by the **XGBoost** model, although other models also demonstrated stable performance. Evaluation is conducted on both Cross-Validation and Validation sets.

### Evaluation Metrics:
- R² Score
- Adjusted R² Score
- Root Mean Squared Log Error (RMSLE)
- Mean Absolute Error (MAE)
- F-statistic

## Feature Engineering/Selection
- First, I reviewed the data and removed columns that contained more than 80% NaN values
- Then, I filled the remaining NaN values in columns with appropriate data (mode of the specific column)
- Next, I removed outliers from the data, as we know that such points have a significant impact on linear models
- Then, I removed columns that contained 97% identical values, as they wouldn't provide valuable information
- The main challenge was converting object-type data to numerical format. For this, I used **WOEEncoder** and converted binary (columns with two distinct values) and non-binary columns to numerical data in different ways
- Finally, I identified columns that were highly correlated with each other and removed one of them to reduce the final feature count.

## Training
- At this stage, I tested 5 models:
    - **LinearRegression**
    - **RandomForest**
    - **XGBoost**
    - **Lasso**
    - **Ridge**
- For hyperparameter optimization, I used a combination of **KFold** and **GridSearchCV**
- As mentioned above regarding the parameters I focused on, I selected the final model based on these parameters

## MLflow Tracking
| Model           | Training R² | Validation R² | Validation RMSE | MAE       | MSLE   | F-statistic | p-value | MLflow Run |
|-----------------|-------------|---------------|------------------|-----------|--------|-------------|---------|------------|
| [LinearRegression](https://dagshub.com/skara-21/Assignment1_ARD.mlflow/#/experiments/0/runs/f157cbfec6064388be772ef3f80880e1) | 0.8625      | 0.8390        | 35136.9262       | 20857.5339 | 0.0260 | 108.09      | 0.0000  | 🔗 |
| [RandomForest](https://dagshub.com/skara-21/Assignment1_ARD.mlflow/#/experiments/0/runs/7b57110eac71412a8bdefec9949775e1) | 0.9791      | 0.8898        | 29069.5874       | 17043.7247 | 0.0215 | -           | -       | 🔗 |
| [XGBoost](https://dagshub.com/skara-21/Assignment1_ARD.mlflow/#/experiments/0/runs/1a3db518053d4ae3a34b4855a4b4f46b) | 0.9814      | 0.9142        | 25647.3774       | 16045.6185 | 0.0181 | -           | -       | 🔗 |
| [Lasso](https://dagshub.com/skara-21/Assignment1_ARD.mlflow/#/experiments/0/runs/2c63b09143b64228be186facb2bd63ce) | 0.8625      | 0.8393        | 35112.4377       | 20832.8629 | 0.0259 | 108.09      | 0.0000  | 🔗 |
| [Ridge](https://dagshub.com/skara-21/Assignment1_ARD.mlflow/#/experiments/0/runs/fcf2dc41f19445fd9151a45e5513c960) | 0.8621      | 0.8399        | 35038.1316       | 20721.6985 | 0.0258 | 107.74      | 0.0000  | 🔗 |

## Best Model: **XGBoost**
