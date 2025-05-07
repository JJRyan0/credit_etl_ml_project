# Credit Risk Pipeline: End-to-End Data + ML Project

This project simulates a real-world credit scoring pipeline, built with dbt, Python, Postgres, and scikit-learn to demonstrate data engineering and MLOps capabilities.

## Tech Stack
- dbt for data modeling (staging + transformations)

- ![image](https://github.com/user-attachments/assets/b369293d-4bb9-4523-966e-b8e4c0bab1c6)

- Postgres as the data source and target
- Python for ML (scikit-learn, pandas)
- FastAPI for real-time scoring (optional)
- MLflow (optional) for model versioning

## 📊 Features
- Raw → staged → final data model using dbt
- Model trained to predict `is_default`
- k-fold cross-validation with F1 score evaluation
- Batch or API inference pipeline
- Clean modular structure for easy deployment

## 🧪 To Reproduce
1. Set up Postgres and run seed scripts
2. Run `dbt run` to create staging + final tables
3. Use `notebooks/train_model.ipynb` to train
4. Run `inference/score.py` to generate predictions

