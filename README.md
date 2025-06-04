# Telecom Customer Churn Prediction

This project focuses on predicting customer churn in the telecom industry using a machine learning model. The model is built using the Random Forest algorithm and trained on a Kaggle dataset. The project also includes a web application developed using Flask, allowing users to input customer data and get predictions on whether a customer is likely to churn or not.

## Table of Contents

- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Web Application](#web-application)
- [How to Run](#how-to-run)
- [Results](#results)

## Dataset

The dataset used in this project is sourced from Kaggle and contains customer data from a telecom company. It includes various features like customer tenure, monthly charges, total charges, and several categorical features like gender, contract type, and payment method.

- **Filename:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size:** ~10,000 rows and 21 columns

## Data Preprocessing

### Steps Taken:
1. **Handling Missing Values:**
   - Converted the `TotalCharges` column to numeric and handled missing values by removing rows with missing data.
   
2. **Feature Engineering:**
   - Grouped the `tenure` column into bins representing different time periods.
   - Dropped irrelevant columns like `customerID` and `tenure`.

3. **Encoding Categorical Variables:**
   - Converted categorical variables into dummy variables using `pd.get_dummies()`.

## Exploratory Data Analysis (EDA)

The EDA involved visualizing the distribution of features and their relationship with the target variable (`Churn`):

- **Target Variable Distribution:** Analyzed the distribution of churned vs. non-churned customers.
- **Correlation Analysis:** Heatmaps and bar charts were used to identify correlations between features and churn.
- **Univariate and Bivariate Analysis:** Plots were generated to explore the relationship between individual features and churn.

## Model Building

### Steps:
1. **Train-Test Split:** Split the data into training and testing sets with an 80-20 split.
2. **Model Selection:** Used a Random Forest Classifier for its ability to handle large datasets and provide feature importance.
3. **Handling Imbalanced Data:** Used SMOTEENN to handle the imbalanced nature of the dataset.

### Model Parameters:
- **n_estimators:** 100
- **max_depth:** 6
- **min_samples_leaf:** 8
- **random_state:** 100

## Model Evaluation

The model was evaluated using metrics such as:

- **Accuracy**
- **Recall**
- **Precision**
- **Confusion Matrix**

Results were compared before and after applying SMOTEENN to balance the dataset.

## Web Application

An interactive web interface was developed using Flask, enabling users to enter customer details and instantly receive churn predictions. The backend loads the pre-trained Random Forest model and returns both the prediction and its associated probability.

#### Project Structure:
- **app.py:** Flask server and prediction logic.
- **home.html:** User-facing HTML form for data entry.

## Running the Application

### Requirements:
- Python 3.x
- Libraries: `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `imblearn`, `flask`, `joblib`

### Instructions:
1. Download or clone this repository:
   ```shell
   git clone https://github.com/mannatsingla22/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```
2. Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```
3. Launch the web server:
   ```shell
   python app.py
   ```
4. Open your browser and go to `http://localhost:5000/` to use the app.

### Results:
The model achieved a satisfactory level of accuracy and recall, especially after handling the imbalanced dataset using SMOTEENN. The web application provides an easy-to-use interface for predicting customer churn with a clear confidence score.
