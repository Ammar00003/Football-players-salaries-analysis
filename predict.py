import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import joblib
import argparse

def predict_salary(age, position, nationality, team):
    model = joblib.load('model/salary_predictor.joblib')
    sample = pd.DataFrame({
        'Age': [age],
        'Position': [position],
        'Nationality': [nationality],
        'Team': [team]
    })
    return model.predict(sample)[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', type=int, required=True)
    parser.add_argument('--position', required=True)
    parser.add_argument('--nationality', required=True)
    parser.add_argument('--team', required=True)
    args = parser.parse_args()
    
    prediction = predict_salary(args.age, args.position, args.nationality, args.team)
    print(f"Predicted salary: £{prediction:,.2f}")