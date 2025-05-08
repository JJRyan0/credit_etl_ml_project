import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import issparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
import joblib


#Load Data from PostgreSQL
import pandas as pd
from sqlalchemy import create_engine

db_config = {
    "user": "EXAMPLE",
    "password": "EXAMPLE",
    "host": "localhost",
    "port": 5432,
    "database": "credit_etl"
}

engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

#load data to data frame

query1 = "select * from raw.stg_credit_data"
query2 = "select * from raw.credit_score_summary"

credit_data = pd.read_sql(query1, engine)
#credit_score_summary = pd.read_sql(query2, engine)

x = credit_data.drop(columns= ['is_default', 'customer_id'])
y = credit_data['is_default']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

#custom transformer to convert sparse matrix to dense array
class SparseToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y='none'):
        return self #nothing to learn so return self
        
    def transform(self, x):
        if issparse(x):
            return x.toarray() # convert sparse to dense
        return x # already dense return



categorical_cols = ['job', 'housing', 'purpose']
numerical_cols = ['age', 'duration','credit_amount']

# Calculate the class weight ratio for imbalanced classes
class_weight_ratio = len(y_train) / (2 * y_train.value_counts().min())


#define the column transformer with OneHotEncoder
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='auto')

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('to_dense', SparseToDenseTransformer()),
    ('smote',SMOTE(random_state=42)),#SMOTE to handle class imbalance
    ('clf', XGBClassifier(eval_metric='auc', random_state=42, scale_pos_weight=class_weight_ratio))
])


# Define the scoring method for cross-validation (AUC in this case)
scoring = make_scorer(roc_auc_score)

param_grid = {
    'clf__n_estimators': [50, 100, 200, 300],
    'clf__max_depth': [3],
    'clf__learning_rate': [0.1, 0.01],
    'clf__subsample': [0.8],
    'clf__reg_alpha': [0, 0.1],  # L1 regularization
    'clf__reg_lambda': [1, 10]      # L2 regularization
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                           scoring=scoring, cv=cv, verbose=1, n_jobs=-1)

grid_search.fit(x_train, y_train)  # optional, suppress per-iteration output

best_params = grid_search.best_params_
print('Best Params:', best_params)
print("Best AUC:", grid_search.best_score_)


best_model = grid_search.best_estimator_
y_pred_proba = best_model.predict_proba(x_test)[:,1]
test_auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC on test set: {test_auc:.4f}")

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model_cr_pipeline.pkl')
