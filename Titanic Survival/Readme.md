# Titanic Survival Prediction

## Overview
This project is a machine learning solution to the classic Titanic survival prediction problem. Using the Titanic dataset from Kaggle, we develop a predictive model to determine the likelihood of a passenger surviving based on various features such as age, sex, fare, and class.

## Dataset
The dataset used for this project is the Titanic dataset from Kaggle, which contains information on passengers, including their demographics and survival status.

### Features
- **PassengerId**: Unique identifier for each passenger
- **Survived**: Target variable (1 = survived, 0 = did not survive)
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Name**: Passenger name
- **Sex**: Gender (male/female)
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Ticket fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Approach
1. **Data Preprocessing**
   - Handling missing values using mean imputation
   - Encoding categorical variables (Sex and Embarked) with One-Hot Encoding
   - Feature selection by dropping irrelevant columns (Embarked, Name, Ticket, Cabin, Sex, N)
   - Feature scaling using StandardScaler
2. **Exploratory Data Analysis (EDA)**
   - Distribution analysis of survival and class features
   - Stratified sampling to maintain class balance
3. **Model Selection & Training**
   - Used a **Random Forest Classifier** for prediction
   - Tuned hyperparameters using **GridSearchCV**
   - Evaluated model using accuracy metric
4. **Prediction & Submission**
   - Optimized model performance
   - Generated final predictions for Kaggle submission

## Results
- The best-performing **Random Forest Classifier** was selected through hyperparameter tuning.
- The final model was tested and applied to the test dataset for submission.

## Dependencies
To run this notebook, install the required dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage
1. Clone the repository and navigate to the project directory.
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook titanic-survival.ipynb
   ```
3. Follow the steps in the notebook to preprocess the data, train models, and generate predictions.
4. The final predictions will be saved as `predictions.csv`.

## Future Improvements
- Hyperparameter tuning for better performance
- Feature engineering to improve predictive power
- Experimenting with deep learning models

## Acknowledgments
- Kaggle Titanic Dataset: [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

## Author
- Sai Sampath Ayalasomayajula

