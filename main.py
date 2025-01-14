from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Iterator, Sequence


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the CustomScaler class
class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)  # Assigning the scaler object to the instance variable self.scaler
        self.columns = columns  # Assigning the list of column names to the instance variable self.columns
        self.mean_ = None  # Initialize mean attribute to None for later calculation
        self.var_ = None  # Initialize var attribute to None for later calculation

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)  # Fit the scaler object to the assigned list of columns among the X columns
        self.mean_ = np.mean(X[self.columns], axis=0)  # Compute the mean of the fitted columns column-wise and assign it to self.mean
        self.var_ = np.var(X[self.columns], axis=0)  # Compute the variance of the fitted columns column-wise and assign it to self.var
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns  # Specify the initial column order
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns, index=X.index)  # Transform the selected columns
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]  # Select columns not in self.columns
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]  # Return the concatenation of the two DataFrames in specified column order


# Initialize FastAPI app
app = FastAPI(title="Simple Model Deployment",
              description="Logistic Regression Model API",
              version="1.0")

# Load the model
model = joblib.load('./models/model.pkl')

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve static files (e.g., CSS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Schema for Employee Data
class EmployeeData(BaseModel):
    Reason_for_Absence: int = Field(..., description="range= 1-28")
    Month: int = Field(..., description="range= 1-12")
    Transportation_Expense: int = Field(..., description="expenses in dollars")
    Age: int = Field(..., description="range= 18-65")
    Body_Mass_Index: int = Field(..., description="range= 18-40")
    Education: int = Field(..., description="0 or 1")
    Children: int = Field(..., description="Number of children")
    Pets: int = Field(..., description="Number of Pets")

# Output schema
class OutputDataModel(BaseModel):
    Prediction: str
    Probability: str

# Preprocess input to match model expectations
def preprocess_input(data: EmployeeData):
    # Directly map Reason_for_Absence to 1, 2, 3, or 4 based on the input
    if 1 <= data.Reason_for_Absence <= 14:
        reason_1 = 1
        reason_2 = reason_3 = reason_4 = 0

    elif 15 <= data.Reason_for_Absence <= 17:
        reason_2 = 1
        reason_1 = reason_3 = reason_4 = 0

    elif 18 <= data.Reason_for_Absence <= 21:
        reason_3 = 1
        reason_1 = reason_2 = reason_4 = 0

    elif 22 <= data.Reason_for_Absence <= 28:
        reason_4 = 1
        reason_1 = reason_2 = reason_3 = 0
        
    else:
        reason_1 = reason_2 = reason_3 = reason_4 = 0  # Invalid reason

    # Create a final input dictionary
    final_data = {
        "Reason 1": reason_1,
        "Reason 2": reason_2,
        "Reason 3": reason_3,
        "Reason 4": reason_4,
        "Month Value": data.Month,
        "Transportation Expense": data.Transportation_Expense,
        "Age": data.Age,
        "Body Mass Index": data.Body_Mass_Index,
        "Education": data.Education,
        "Children": data.Children,
        "Pets": data.Pets
    }

    return pd.DataFrame([final_data])

# Endpoint to predict and return both JSON and HTML response
# Endpoint to predict using JSON data
@app.post("/predict/json", response_model=OutputDataModel)
async def predict_json(request: Request, data: EmployeeData):
    return await process_prediction(request, data, response_type="json")

# Endpoint to predict using form data
@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request, 
                        Reason_for_Absence: int = Form(...), 
                        Month: int = Form(...), 
                        Transportation_Expense: int = Form(...), 
                        Age: int = Form(...), 
                        Body_Mass_Index: int = Form(...), 
                        Education: int = Form(...), 
                        Children: int = Form(...), 
                        Pets: int = Form(...)):
    
    # Construct the EmployeeData object from form fields
    data = EmployeeData(
        Reason_for_Absence=Reason_for_Absence,
        Month=Month,
        Transportation_Expense=Transportation_Expense,
        Age=Age,
        Body_Mass_Index=Body_Mass_Index,
        Education=Education,
        Children=Children,
        Pets=Pets
    )
    return await process_prediction(request, data, response_type="html")



# Logic for both endpoints
async def process_prediction(request: Request, data: EmployeeData, response_type: str):
    try:
        input_data = preprocess_input(data)

        # Scalling specific features
        columns_to_scale = ['Transportation Expense', 'Age', 'Body Mass Index', 'Children', 'Pets', 'Month Value']
        absenteeism_scaler = CustomScaler(columns_to_scale)
        absenteeism_scaler.fit(input_data)
        scaled_input_data = absenteeism_scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input_data)
        probability = model.predict_proba(scaled_input_data)[0][1]

        # Prepare response model
        output_data = OutputDataModel(
            Prediction="Excessively Absent" if prediction[0] == 1 else "Moderately to Not Absent",
            Probability=f"The probability of being excessively absent is {round(probability, 2)}"
        )

        if response_type == "json":
            # Return JSON response
            return JSONResponse(content=output_data.model_dump())
        
        # Return HTML response using templates
        return templates.TemplateResponse(request, "result.html", {"prediction": output_data.Prediction, "probability": output_data.Probability})
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in prediction.")

# Endpoint to serve the form
@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse(request, "form.html")