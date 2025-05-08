

#Libaries and dependencies
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
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

###=======Load Data from PostgreSQL=============##
import pandas as pd
from sqlalchemy import create_engine

db_config = {
    "user": "INSERT",
    "password": "INSERT",
    "host": "localhost",
    "port": 5432,
    "database": "credit_etl"
}

engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

#load data from PGsql DB to a DataFrame
query = "select * from raw.raw_customer_default_payment"

credit_data = pd.read_sql(query, engine)

##=======Pipline - Credit Risk Model - XGboost Classification - Credit Best Model V1===========#

#create  x and y as target variable
x = credit_data.drop(columns= ['default_payment_next_month', 'id'])
y = credit_data['default_payment_next_month']

#train test split with test size 20%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)


#Define the numerical cols for the preprocessor activities
numerical_cols = credit_data

# Calculate the scale weight ratio for imbalanced classes
counter = Counter(y_train) 
scale_weight_ratio = counter[0] / counter[1]

#define the column transformer to scale the data using standard method
preprocessor = ColumnTransformer(transformers=[
    #('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('num', StandardScaler(), slice(0, x.shape[1]))
])

#Apply SMOTE to balance all classes to match the majority class.
smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='auto')

#Kfold stratification method
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('smote',SMOTE(random_state=42)),#SMOTE to handle class imbalance
    #('selection', SelectKBest(score_func = f_classif, k=15)),
    ('clf', XGBClassifier(eval_metric='auc', random_state=42, scale_pos_weight=scale_weight_ratio))
])


# Define the scoring method for cross-validation (AUC in this case)
scoring = make_scorer(roc_auc_score)

param_grid = {
    'clf__n_estimators': [50, 100, 200, 300],
    'clf__max_depth': [3,5],
    'clf__learning_rate': [0.01, 0.05],
    'clf__subsample': [0.8],
    'clf__reg_alpha': [0, 0.3, 0.7],  # L1 regularization
    'clf__reg_lambda': [1, 10]      # L2 regularization
}

#Fitting 10 folds for each of 96 candidates, totalling 960 fits
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                           scoring=scoring, cv=cv, verbose=1, n_jobs=-1)

grid_search.fit(x_train, y_train)  # optional, suppress per-iteration output

#gwt best params & scores for use in predict test probabilities 
best_params = grid_search.best_params_
print('Best Params:', best_params)
print("Best AUC:", grid_search.best_score_)

#usiing the best estimator from grid search predict on test set
credit_best_model = grid_search.best_estimator_
y_pred_proba = credit_best_model.predict_proba(x_test)[:,1]
test_auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC on test set: {test_auc:.4f}")

joblib.dump(best_model, 'credit_best_model_V1.pkl')

