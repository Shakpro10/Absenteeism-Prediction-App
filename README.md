## Absenteeism Prediction App

This application predicts the likelihood of employee absenteeism in an organization. The model determines whether a worker is "excessively absent" or "moderately to not absent" based on various input features.
________________________________________


### Project Overview

The goal of this app is to help organizations forecast and mitigate absenteeism, optimizing workforce management and productivity. By analysing key employee attributes, the model can predict if an individual is likely to be excessively absent (defined as being absent for more than 3 hours).
________________________________________


### Prediction Categories:

•	Excessively Absent – Probability of high absenteeism.
•	Moderately to Not Absent – Probability of low or no absenteeism.
________________________________________


### Features Used in Prediction

1.	Reason for Absence – Categorized into 4 groups:
    •	1-14 – Disease-based reasons.
    •	15-18 – Pregnancy-related reasons.
    •	19-21 – Poison-based reasons.
    •	22-28 – Other reasons.
2.	Transportation Expense – Cost of transportation to work (in dollars).
3.	Month – Month of the year the absence occurred.
4.	Age – Employee's age.
5.	Body Mass Index (BMI) – BMI of the employee.
6.	Number of Children – Number of children the worker has.
7.	Number of Pets – Number of pets the worker owns.
________________________________________


### Project Structure

#### Absenteeism Prediction App/

###### datasets/
###### Absenteeism_data.csv
###### Absenteeism_preprocessed.csv

#### Models/
###### model.pkl

#### notebook/
###### Preprocessing exessively and moderately absent employees datasets.ipynb
###### Applying Machine Learning to the Absenteeism Dataset.ipynb

#### templates/
###### form.html
###### result.html

#### static/
##### css/
###### styles.css
##### images/
###### LogisticRegression.jpg

#### .github/
##### workflows/
###### deploy.yml

#### model.py
#### main.py
#### test.py
#### Dockerfile
#### Dockerrun.aws.json
#### heroku.yml
#### requirements.txt
#### runtime.txt
#### .gitignore
#### .dockerignore
#### README.md


### How the App Works

1.	Users provide the necessary input features through a form on the web app.
2.	The app uses a Logistic Regression model to predict absenteeism based on the provided data.
3.	Results are displayed on a result page, indicating the probability of excessive absenteeism.
________________________________________


### API Endpoints

Endpoint	Method	Description
/	        GET	    Renders the form for predictions.
/predict	POST	Accepts form data and makes predictions.
________________________________________


### Example Prediction

Inputs Example:
    •	Reason for Absence: 10
    •	Transportation Expense: 300
    •	Month: 5
    •	Age: 35
    •	BMI: 28
    •	Number of Children: 2
    •	Number of Pets: 1

Prediction Output: Excessively Absent – 72% probability
________________________________________


### Why Logistic Regression?

After training the dataset with various machine learning models, Logistic Regression emerged as the top performer for this project. Here are the reasons why Logistic Regression was ultimately chosen:
    •	Simplicity – Logistic Regression is easy to implement and computationally efficient, making it ideal for binary classification problems.
    •	Interpretability – The model provides clear insights into the relationship between features and predictions, making results easier to understand and explain.
    •	Effectiveness – Logistic Regression consistently delivered high accuracy during testing, outperforming other models such as Random Forest, XGBoost, KNN, Decision Trees and Support Vector Machines for this specific dataset.
    •	Binary Classification – Since the goal is to classify employees into two categories (excessive or moderate absenteeism), Logistic Regression aligns well with the project's objectives.


### Key Features

•	Data Preprocessing – The project includes detailed preprocessing steps to clean and transform the absenteeism dataset, ensuring the model receives high-quality input.
•	Model Deployment – The app is containerized using Docker and deployable on platforms like Heroku and AWS.
•	Web Interface – A simple web interface allows users to input employee data and receive predictions on absenteeism likelihood.

### Notebooks

The project includes two Jupyter notebooks located in the notebook/ folder:
    1.	Preprocessing exessively and moderately absent employees datasets.ipynb – This notebook focuses on data cleaning, feature engineering, and dataset balancing.
    2.	Applying Machine Learning to the Absenteeism Dataset.ipynb – This notebook implements various models and evaluates their performance, culminating in the selection of Logistic Regression.

________________________________________


### Tech Stack

    •	Backend – FastAPI
    •	Frontend – HTML/CSS
    •	Model – Logistic Regression (Scikit-Learn)
    •	Deployment – Heroku, AWS
    •	Containerization – Docker
________________________________________


### Installation and Setup

To run the app locally:
Bash code
"""Clone the repository"""
git clone https://github.com/Shakpro10/Abenteeism-Prediction-App.git  

"""Navigate into the project directory"""
cd Abenteeism-Prediction-App  

"""Install dependencies"""
pip install -r requirements.txt  

"""Run the FastAPI app"""
uvicorn main:app --reload  
________________________________________


### Deployment

The app is deployed on Heroku and accessible through the following URL: https://absenteeism-predictor-95f9fbe5a2b8.herokuapp.com/
________________________________________


### Testing

To run tests:
Bash code
pytest test.py  
________________________________________


### Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request.
________________________________________