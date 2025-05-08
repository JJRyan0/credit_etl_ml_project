from fastapi impoty FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# load the trained model
model = joblib.load('credit_best_model_V1.pkl')

# Create FastAPI app
app = FastAPI()

class CreditData(BaseModel):
    LIMIT_BAL: Optional[int]
    SEX: Optional[int]
    EDUCATION: Optional[int]
    MARRIAGE: Optional[int]
    AGE: Optional[int]
    PAY_0: Optional[int]
    PAY_2: Optional[int]
    PAY_3: Optional[int]
    PAY_4: Optional[int]
    PAY_5: Optional[int]
    PAY_6: Optional[int]
    BILL_AMT1: Optional[int]
    BILL_AMT2: Optional[int]
    BILL_AMT3: Optional[int]
    BILL_AMT4: Optional[int]
    BILL_AMT5: Optional[int]
    BILL_AMT6: Optional[int]
    PAY_AMT1: Optional[int]
    PAY_AMT2: Optional[int]
    PAY_AMT3: Optional[int]
    PAY_AMT4: Optional[int]
    PAY_AMT5: Optional[int]
    PAY_AMT6: Optional[int]

@app.post("/predict")
def predict(payload: CreditData):
    df = pd.DataFrame([payload])
    prediction = model.predict(df)[0]
    return {"is_default": int(prediction)}