# 🩺 Diabetes Prediction using Machine Learning

An end-to-end machine learning project that predicts the diabetes class (`0` or `1`) from patient health information.

The project has three parts:

- `Untitled9.ipynb` — experimentation/EDA and model development
- `train.py` — reproducible model training and evaluation
- `app.py` — Streamlit deployment interface

## Features

- Logistic Regression
- Numerical feature standardization
- Categorical feature One-Hot Encoding
- Train/test split with stratification
- Accuracy, classification report and confusion matrix
- Saved preprocessing + model pipeline
- Interactive Streamlit prediction form

## Dataset

The project uses a diabetes dataset containing:

```text
gender
age
hypertension
heart_disease
smoking_history
bmi
HbA1c_level
blood_glucose_level
diabetes
```

`diabetes` is the target variable.

## Project Structure

```text
diabetes-prediction/
│
├── app.py
├── train.py
├── diabetes_prediction_dataset.csv
├── diabetes_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## How the ML Pipeline Works

```text
Dataset
   ↓
Remove duplicates
   ↓
X / y split
   ↓
Train/Test Split
   ↓
┌─────────────────────────────┐
│ Numerical → StandardScaler  │
│ Categorical → OneHotEncoder │
└─────────────────────────────┘
   ↓
Logistic Regression
   ↓
Evaluation
   ↓
Save complete pipeline
```

The preprocessing and classifier are saved together in `diabetes_model.pkl`. This is important because the exact same transformations used during training must be applied when a new patient is entered.

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This creates:

```text
diabetes_model.pkl
```

### 3. Start the Streamlit app

```bash
streamlit run app.py
```

## GitHub

Initialize the repository:

```bash
git init
git add .
git commit -m "Add diabetes prediction ML project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Streamlit Community Cloud

Push the project to GitHub and create a new Streamlit app.

Set:

```text
Main file: app.py
```

Keep these files in the repository:

```text
app.py
diabetes_model.pkl
requirements.txt
```

The CSV and `train.py` are useful for reproducibility and retraining.

## Why OneHotEncoder?

The categorical features such as `gender` and `smoking_history` are nominal categories. One-Hot Encoding avoids assigning an artificial numerical order to these categories.

The preprocessing is handled by a `ColumnTransformer`, so new patient data goes through the same transformations automatically.

## Important

This is an educational machine learning project. The prediction and probability are model outputs and **must not be used as a medical diagnosis or as a substitute for a qualified healthcare professional**.
