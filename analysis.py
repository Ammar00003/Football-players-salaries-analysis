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

# Create directory for plots
os.makedirs('plots', exist_ok=True)
# Create necessary directories
os.makedirs('plots', exist_ok=True)
os.makedirs('model', exist_ok=True)

# Load and clean data
df = pd.read_csv("data/soccer_salaries.csv")
df['Yearly Salary Numeric'] = df['Yearly Salary'].str.replace('£', '').str.replace(',', '').astype(float)

# Analysis: Top 10 highest-paid players
top_10 = df.nlargest(10, 'Yearly Salary Numeric')[['Player Name', 'Yearly Salary Numeric', 'Team']]

# Visualisation
plt.figure(figsize=(10, 6))
sns.barplot(data=top_10, x='Yearly Salary Numeric', y='Player Name', hue='Team', palette='viridis')
plt.title('Top 10 Highest-Paid Football Players (Yearly Salary)')
plt.xlabel('Yearly Salary (£)')
plt.ylabel('Player Name')
plt.tight_layout()
plt.savefig('plots/top_earners_in_league.png') 
plt.close()

# Compare Salaries by Nationality
plt.figure(figsize=(12, 6))
nationality_avg = df.groupby('Nationality')['Yearly Salary Numeric'].mean().nlargest(10)
sns.barplot(x=nationality_avg.values, y=nationality_avg.index, palette='rocket', hue=nationality_avg.index, legend=False)
plt.title('Top 10 Highest-Paying Nationalities (Average Salary)')
plt.xlabel('Average Yearly Salary (£)')
plt.tight_layout()
plt.savefig('plots/nationality_salaries.png')
plt.close()

# Compare Salaries by Age
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Age', y='Yearly Salary Numeric', alpha=0.6)
plt.title('Salary Distribution by Age')
plt.ylabel('Yearly Salary (£)')
plt.tight_layout()
plt.savefig('plots/age_salaries.png')
plt.close()

# Build Regression Model
# Prepare data
X = df[['Age', 'Position', 'Nationality', 'Team']]
y = df['Yearly Salary Numeric']

# Preprocessing and model pipeline (Linear Regression model)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Position', 'Nationality', 'Team'])
    ],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X, y)

# Example prediction, using a hypothetical brazilian 25 year old united striker
sample_player = pd.DataFrame({
    'Age': [25],
    'Position': ['ST'],
    'Nationality': ['Brazil'],
    'Team': ['manchester-united-f.c.']
})
predicted_salary = model.predict(sample_player)[0]
print(f"Predicted Yearly Salary: £{predicted_salary:,.2f}")

# Save model
joblib.dump(model, 'model/salary_predictor.joblib')
print("Model saved successfully!")
