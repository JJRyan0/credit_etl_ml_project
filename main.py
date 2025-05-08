from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Optional

# load the trained model
model = joblib.load('credit_best_model_V1.pkl')

# Create FastAPI app
app = FastAPI()

class CreditData(BaseModel):
    limit_bal: Optional[int]
    sex: Optional[int]
    education: Optional[int]
    marriage: Optional[int]
    age: Optional[int]
    pay_0: Optional[int]
    pay_2: Optional[int]
    pay_3: Optional[int]
    pay_4: Optional[int]
    pay_5: Optional[int]
    pay_6: Optional[int]
    bill_amt1: Optional[int]
    bill_amt2: Optional[int]
    bill_amt3: Optional[int]
    bill_amt4: Optional[int]
    bill_amt5: Optional[int]
    bill_amt6: Optional[int]
    pay_amt1: Optional[int]
    pay_amt2: Optional[int]
    pay_amt3: Optional[int]
    pay_amt4: Optional[int]
    pay_amt5: Optional[int]
    pay_amt6: Optional[int]

    # This method will automatically convert all field names to lowercase
    class Config:
        alias_generator = lambda x: x.lower()


@app.post("/predict")
def predict(payload: CreditData):
    try:
        # Convert payload to DataFrame
        df = pd.DataFrame([payload.dict()])

        # Debugging: Check DataFrame after payload processing
        print(f"Processed DataFrame: \n{df}")  # Debugging line

        # Access the final estimator (XGBoost model) from the pipeline
        xgb_model = model.named_steps['clf']  # Change 'clf' if your model is named differently in the pipeline
        print(f"XGBoost model retrieved from pipeline: {xgb_model}")  # Debugging line

        # Ensure the input DataFrame columns match the model's expected columns
        model_columns = [
            'limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 
            'pay_5', 'pay_6', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6', 
            'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6'
        ]
        print(f"Model expects columns: {model_columns}")  # Debugging line

        # Ensure the input DataFrame columns match the model's expected columns
        df = df[model_columns]  # Reorder the columns to match the model's expected order
        print(f"Reordered DataFrame: \n{df}")  # Debugging line

        # Make prediction
        prediction = model.predict(df)[0]
        return {"is_default": int(prediction)}

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debugging line
        return {"error": str(e)}
