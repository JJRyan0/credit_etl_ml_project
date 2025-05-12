# Credit Risk Pipeline: End-to-End Data + ML Project

John Ryan - 10 May 2025

This project simulates a real-world credit scoring pipeline, built with dbt, Python, Postgres, and scikit-learn to demonstrate data engineering and MLOps capabilities.

<img width="1111" alt="image" src="https://github.com/user-attachments/assets/9d72641c-a118-4c6d-996d-51ff029e4926" />


## Tech Stack
- dbt for data modeling (staging + transformations) for synthetic data pipeline test (Here: [synthetic data pipeline test.ipynb](https://github.com/JJRyan0/credit_etl_ml_project/blob/main/notebooks/Synthetic%20data%20pipeline%20test.ipynb))
- Postgres as the data source and target 
- __Credit Risk Classifier with XGboost Python(scikit-learn)__ (Pipeline Here:[Credit Default-XGboost Classify Analysis.ipynb)](https://github.com/JJRyan0/credit_etl_ml_project/blob/main/notebooks/Credit%20Default-XGboost%20Classify%20Analysis.ipynb)
- FastAPI for [real-time scoring App](https://github.com/JJRyan0/credit_etl_ml_project/blob/main/main.py)
- MLflow (optional) for [ML model versioning](https://github.com/JJRyan0/credit_etl_ml_project/blob/main/MLflow%20Model%20Versioning.md)

## Features
- Raw → staged → final data model using dbt
- Model trained to predict `is_default` (Here: notebooks/Credit Default-XGboost Classify Analysis.ipynb)
- k-fold cross-validation with F1 score evaluation
- SMOTE - Helps XGBoost focus more on the minority class, improving recall and AUC for imbalanced datasets.
- Batch or API inference pipeline
- Clean modular structure for easy deployment
- [Feature Engineering (In progress for phase 2)](https://github.com/JJRyan0/credit_etl_ml_project/blob/main/model_stage_2/feature_engineering/Credit%20Default%20-%20Feature%20Engineering%20for%20Model%20v2.ipynb) | Recent Payment Ratio, Average Pay Amount, Average Bill Amount.

## To Reproduce
1. Set up Postgres and run seed scripts
2. Run `dbt run` to create staging + final tables
3. Use `notebooks/Credit Default-Pipeline Full.ipynb` to train
4. Run the below to execute real_time predictions with Fast API app


-------------------------------------------------
# Test the predictive model FastAPI App here:

Start up the application once all requirements are installed from requiremnts.txt file

1. clone repo

```bash

% cd /folder/your_root_folder

```
2. launch the application in the terminal
   
```bash

uvicorn main:app --reload

```
3. Navigate to swagger FastAPI UI
   
```html

http://localhost:8000/docs

```

4. In the Swagger UI, click on /predict, then “Try it out” and enter a sample JSON payload like:

```json

{
  "limit_bal": 20000,
  "sex": 1,
  "education": 2,
  "marriage": 1,
  "age": 25,
  "pay_0": 0,
  "pay_2": 0,
  "pay_3": 0,
  "pay_4": 0,
  "pay_5": 0,
  "pay_6": 0,
  "bill_amt1": 3913,
  "bill_amt2": 3102,
  "bill_amt3": 689,
  "bill_amt4": 0,
  "bill_amt5": 0,
  "bill_amt6": 0,
  "pay_amt1": 0,
  "pay_amt2": 689,
  "pay_amt3": 0,
  "pay_amt4": 0,
  "pay_amt5": 0,
  "pay_amt6": 0
}

```

Execute post request to  the trained model application:

![image](https://github.com/user-attachments/assets/2f9c9979-56e7-45f3-b4ac-73827f84eb80)

Prediction output: 

![image](https://github.com/user-attachments/assets/9a5f3cab-7478-4cd5-b3aa-42a8956f97c0)

