# Santander Customer Transaction Prediction

## Overview
This repository contains a machine learning pipeline for the **Santander Customer Transaction Prediction** competition on Kaggle. The goal of this competition is to predict whether a customer will make a transaction using an anonymized dataset provided by Santander Bank.

## Dataset
The dataset consists of numerical features with no feature descriptions. The challenge is to build a model that can effectively distinguish between transaction and non-transaction customers.

- **Train Dataset:** `train.csv` (contains `ID_code`, 200 anonymized numerical features, and a `target` column)
- **Test Dataset:** `test.csv` (contains `ID_code` and 200 anonymized numerical features, but no `target` column)

## Approach

### 1️⃣ Data Preprocessing
- Removed the **ID_code** column as it is not useful for prediction.
- Checked for **missing values** (none found).
- Examined the **class distribution** (imbalanced dataset with more `0s` than `1s`).
- Visualized data with **histograms, boxplots, and correlation matrices** to detect patterns.

### 2️⃣ Outlier Detection
- Used the **3-standard deviation rule** to identify potential outliers.
- Considered handling extreme values to improve model performance.

### 3️⃣ Feature Scaling
- Applied **StandardScaler** to normalize the data (mean = 0, standard deviation = 1).

### 4️⃣ Handling Class Imbalance
- **Downsampled** the majority class (class `0`) to match the minority class (class `1`).
- Alternative methods like **SMOTE** can be explored.

### 5️⃣ Model Training
- Implemented and compared **Logistic Regression, Random Forest, and Naive Bayes classifiers**.
- Used **10-fold cross-validation (CV)** with **AUC (Area Under Curve) as the performance metric**.
- Selected **Gaussian Naive Bayes** as the best-performing model.

### 6️⃣ Model Evaluation
- Evaluated performance on an **80-20 train-test split**.
- Used metrics such as:
  - **AUC (Area Under Curve)**
  - **Accuracy**
  - **Precision**
  - **Recall**

### 7️⃣ Predictions & Kaggle Submission
- Loaded the **Kaggle test dataset** and applied the same preprocessing steps.
- Made predictions using the **trained Naive Bayes model**.
- Created a **submission CSV file** for Kaggle.

## Installation & Usage
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Running the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/santander-prediction.git
   cd santander-prediction
   ```
2. Run the preprocessing and training script:
   ```bash
   python train_model.py
   ```
3. Generate Kaggle submission:
   ```bash
   python generate_submission.py
   ```

## Improvements & Next Steps
- **Avoid Data Leakage:** Standardize only after train-test split.
- **Handle Outliers More Effectively:** Try IQR or Isolation Forest.
- **Better Class Imbalance Handling:** Use **SMOTE instead of downsampling**.
- **Use Hyperparameter Tuning:** Optimize models with GridSearchCV.
- **Try Advanced Models:** Use XGBoost, LightGBM, and ensemble methods.
- **Threshold Tuning:** Adjust the decision threshold using precision-recall tradeoff.

## Author
Sai Sampath Ayalasomayajula

## Acknowledgments
- Kaggle for providing the dataset.
- Santander Bank for hosting the challenge.

## License
This project is open-source under the MIT License.

