# House Prices - Advanced Regression Techniques

## Overview
This project is a solution to the Kaggle competition "House Prices - Advanced Regression Techniques." The goal is to predict house sale prices using various features available in the dataset. The XGBoost model was implemented for regression.

## Steps Followed

### 1. Getting Started

#### Loading the Data
- The training and test datasets were loaded using `pd.read_csv`.
- A backup of the original data (`main_df`) was created to prevent data loss.

#### Handling Missing Values
- **Numerical Columns**: Missing values were replaced with the mean to maintain numerical stability.
- **Categorical Columns**: Missing values were replaced with the mode (most frequent value).
- **Dropped Columns**: Columns with excessive missing values (e.g., `Alley`, `PoolQC`) and non-predictive features (`Id`) were removed.

#### Encoding Categorical Variables
- Used **one-hot encoding** (`pd.get_dummies`) to convert categorical features into numerical format for machine learning compatibility.

#### Combining Train and Test Data
- To ensure consistent one-hot encoding across datasets, train and test data were concatenated before encoding.

### 2. Building the Model

#### Train-Test Split
- The dataset was split into features (`X_train`) and target variable (`y_train`).
- A validation set (`X_val`, `y_val`) was created to evaluate model performance on unseen data.

#### Using XGBoost
- **Why XGBoost?**
  - Efficient and optimized for tabular data.
  - Handles missing values automatically.
  - Utilizes an ensemble of decision trees to enhance accuracy.
  - Includes regularization techniques to prevent overfitting.
- Implemented using `XGBRegressor` since the task is regression-based.

#### Random State = 42
- Ensures reproducibility of results.

#### Model Training
- The model was trained using `classifier.fit(X_train, y_train)`.

#### Validation
- Predictions (`y_pred`) were generated for the validation set.
- Evaluated using **Root Mean Squared Error (RMSE)**.

### 3. Final Steps

#### Predictions
- The trained model was used to predict house prices for the test dataset (`df_test`).

#### Submission File
- A CSV file (`submission.csv`) was created containing `Id` and `SalePrice` columns for Kaggle submission.

## Dependencies
- Python
- Pandas
- NumPy
- XGBoost
- Scikit-learn

## Running the Project
1. Install the required libraries:
   ```bash
   pip install pandas numpy xgboost scikit-learn
   ```
2. Run the script to preprocess data, train the model, and generate predictions.
   ```bash
   python script.py
   ```
3. Submit `submission.csv` to Kaggle.

## Author
- Sai Sampath Ayalasomayajula

## Acknowledgments
- Kaggle for providing the dataset.
- XGBoost for the powerful gradient boosting algorithm.
- Scikit-learn for preprocessing and evaluation tools.


