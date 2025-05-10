![image](https://github.com/user-attachments/assets/8aa5d741-552b-4a84-b19e-d59ac307f5c8)


# 1. Install MLflow

```bash

pip install mlflow

```
# 2. Launch MLflow UI

```bash

% mlflow ui --port 5000

```
# 3. Log and register current version of the model

```python

import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")

#log model with flow
with mlflow.start_run() as run:
    mlflow.sklearn.log_model('credit_best_model_V1.pkl', artifact_path= 'credit_best_model_V1')
    #print(f"Run ID: {run.info.run_id}")

#register model

result = mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/credit_best_model_V1",
    name="credit_score_model"
)

```

# 4. Check the model is registered as version 1

![image](https://github.com/user-attachments/assets/f10b1a4c-b93f-4398-b528-25a6dde8b9b8)
