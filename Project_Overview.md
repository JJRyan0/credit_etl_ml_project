# Credit Risk Pipeline: End-to-End Data + ML Project

This project simulates a real-world credit scoring pipeline, built with dbt, Python, Postgres, and scikit-learn to demonstrate data engineering and MLOps capabilities.

## Tech Stack
- dbt for data modeling (staging + transformations) for synthetic data pipeline test (Here: synthetic data pipeline test.ipynb)
- Postgres as the data source and target 
- Python for ML (scikit-learn, XGboost, pandas) (Here: notebooks/Credit Default-XGboost Classify Analysis.ipynb)
- FastAPI for real-time scoring
- MLflow (optional) for model versioning

## Features
- Raw → staged → final data model using dbt
- Model trained to predict `is_default` (Here: notebooks/Credit Default-XGboost Classify Analysis.ipynb)
- k-fold cross-validation with F1 score evaluation
- SMOTE - Helps XGBoost focus more on the minority class, improving recall and AUC for imbalanced datasets.
- Batch or API inference pipeline
- Clean modular structure for easy deployment

## To Reproduce
1. Set up Postgres and run seed scripts
2. Run `dbt run` to create staging + final tables
3. Use `notebooks/Credit Default-Pipeline Full.ipynb` to train
4. Run `inference/score.py` to generate predictions


-------------------------------------------------
# Test the predictive model:

Start up the application once all requirements are installed from requiremnts.txt file

```bash

uvicorn main:app --reload

```bash

http://localhost:8000/docs

```


In the Swagger UI, click on /predict, then “Try it out” and enter a sample JSON payload like:

```json

{
  "LIMIT_BAL": 20000,
  "SEX": 1,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 25,
  "PAY_0": 0,
  "PAY_2": 0,
  "PAY_3": 0,
  "PAY_4": 0,
  "PAY_5": 0,
  "PAY_6": 0,
  "BILL_AMT1": 3913,
  "BILL_AMT2": 3102,
  "BILL_AMT3": 689,
  "BILL_AMT4": 0,
  "BILL_AMT5": 0,
  "BILL_AMT6": 0,
  "PAY_AMT1": 0,
  "PAY_AMT2": 689,
  "PAY_AMT3": 0,
  "PAY_AMT4": 0,
  "PAY_AMT5": 0,
  "PAY_AMT6": 0

}
```
