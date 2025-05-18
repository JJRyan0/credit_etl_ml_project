import os
import logging
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, roc_auc_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Custom transformer
class SparseToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray() if issparse(X) else X

def load_data():
    try:
        db_config = {
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "database": os.getenv("DB_NAME")
        }
        engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@"
                               f"{db_config['host']}:{db_config['port']}/{db_config['database']}")
        query = "SELECT * FROM raw.stg_credit_data"
        data = pd.read_sql(query, engine)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error("Error loading data from database", exc_info=True)
        raise

def build_pipeline(categorical_cols, numerical_cols, class_weight_ratio):
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])

    return Pipeline(steps=[
        ('preprocess', preprocessor),
        ('to_dense', SparseToDenseTransformer()),
        ('smote', SMOTE(random_state=42)),
        ('clf', XGBClassifier(eval_metric='auc', random_state=42, scale_pos_weight=class_weight_ratio))
    ])

def main():
    data = load_data()
    X = data.drop(columns=['is_default', 'customer_id'])
    y = data['is_default']

    categorical_cols = ['job', 'housing', 'purpose']
    numerical_cols = ['age', 'duration', 'credit_amount']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    class_weight_ratio = len(y_train) / (2 * y_train.value_counts().min())
    pipeline = build_pipeline(categorical_cols, numerical_cols, class_weight_ratio)

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3],
        'clf__learning_rate': [0.1],
        'clf__subsample': [0.8],
        'clf__reg_alpha': [0, 0.1],
        'clf__reg_lambda': [1, 10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    grid_search = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=1)

    try:
        logging.info("Starting model training...")
        grid_search.fit(X_train, y_train)
        logging.info("Model training complete.")
    except Exception as e:
        logging.error("Training failed", exc_info=True)
        return

    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Test AUC: {auc:.4f}")

    joblib.dump(best_model, "best_model_cr_pipeline.pkl")
    logging.info("Model saved to disk.")

if __name__ == "__main__":
    main()
