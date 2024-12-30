# House Prices Prediction

This project aims to predict house prices based on various features using machine learning. We utilized a RandomForestRegressor model to build and evaluate the prediction model.

## Project Overview

We used the "House Prices - Advanced Regression Techniques" dataset from Kaggle. The task was to predict house prices based on different features such as square footage, number of rooms, and other factors.

## Steps and Workflow

### 1. Data Loading

- **Train Data**: `house_prices_data/train.csv`
- **Test Data**: `house_prices_data/test.csv`

The datasets contain both training data and test data. The training data contains a target variable (`SalePrice`), while the test data does not.

### 2. Data Preprocessing

- **Handling Missing Values**: We filled missing values in the dataset using imputation techniques, such as RandomForest imputation for numerical values and mode imputation for categorical values.
- **Feature Engineering**: We performed one-hot encoding for categorical features to prepare the data for the model. We also concatenated the train and test datasets for consistent encoding.

### 3. Model Training

- We used **RandomForestRegressor** from `sklearn` as the model to predict the house prices.
- The training data was split into **training** and **validation** sets using an 80/20 split.
- The model was trained using the training data, and hyperparameters such as `n_estimators` were set to 100.

### 4. Model Evaluation

- We evaluated the model using various metrics:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R-squared (R²)**

  The model performed well, achieving an R² of approximately **0.89**, which indicates that 89% of the variance in house prices was explained by the model.

### 5. Predictions

- We generated predictions using the test data and saved them in a CSV file named `submission.csv` containing two columns: `ID` and `SalePrice`.

### 6. Model Saving

- The trained model was saved using `joblib` to a file named `house_price_model.pkl` for future use in deployment or inference.

## Files

- **train.csv**: The training data containing features and target variable (`SalePrice`).
- **test.csv**: The test data without the target variable (`SalePrice`).
- **submission.csv**: The final output with predicted house prices (`SalePrice`).
- **house_price_model.pkl**: The saved trained model.
- **upd_train.csv**: Preprocessed and encoded training data.
- **upd_test.csv**: Preprocessed and encoded test data.

## Future Work

- **Model Improvement**: Further improvements could include hyperparameter tuning, trying other machine learning algorithms like Gradient Boosting, or adding new features to the dataset.
- **Deployment**: The trained model can be deployed to a web service to provide predictions for new data.

## How to Run the Project

1. Clone the repository.
2. Install necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training and prediction script:
   ```bash
   python house_price_prediction.py
   ```

## Dependencies

- pandas
- numpy
- scikit-learn
- joblib

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



