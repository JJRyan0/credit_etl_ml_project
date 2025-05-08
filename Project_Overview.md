# Credit Risk Pipeline: End-to-End Data + ML Project

This project simulates a real-world credit scoring pipeline, built with dbt, Python, Postgres, and scikit-learn to demonstrate data engineering and MLOps capabilities.

## Tech Stack
- dbt for data modeling (staging + transformations) for synthetic data pipeline test (Here: Credit - synthetic data pipeline test.ipynb)
- Postgres as the data source and target 
- Python for ML (scikit-learn, XGboost, pandas) (Here: credit_etl_ml_project/notebooks/Credit Default-XGboost Classify Analysis.ipynb)
- FastAPI for real-time scoring ()
- MLflow (optional) for model versioning

## Features
- Raw → staged → final data model using dbt
- Model trained to predict `is_default` (Here: credit_etl_ml_project/notebooks/Credit Default-XGboost Classify Analysis.ipynb)
- k-fold cross-validation with F1 score evaluation
- SMOTE - Helps XGBoost focus more on the minority class, improving recall and AUC for imbalanced datasets.
- Batch or API inference pipeline
- Clean modular structure for easy deployment

## To Reproduce
1. Set up Postgres and run seed scripts
2. Run `dbt run` to create staging + final tables
3. Use `notebooks/Credit Default-Pipeline Full.ipynb` to train
4. Run `inference/score.py` to generate predictions

