# Football Players Salaries Analysis  

A data science project analysing Premier League player salaries using Python, with automated visualisations and a predictive ML model.

## Project Overview
This project analyses:
- Top earning players and teams
- Salary distributions by nationality and age
- A machine learning model to predict player salaries

## Key Features
- Visualisations: Automated plot generation (top earners, age/nationality trends)
- Machine Learning: Salary prediction model (Linear Regression) 
- Reproducibility: Full dependency management via `requirements.txt`

## Technical Stack
- Language: Python
- Libraries: Pandas, Matplotlib, Seaborn, Scikit-learn
- Tools: Git, GitHub, GitPod

## Sample Output
![Top Earners Plot](plots/top_earners.png)  
*(Example visualisation from the analysis)*

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis (generates plots and model)
python analysis.py

# Make predictions (example)
python predict.py --age 25 --position ST --nationality Brazil --team arsenal-f.c.
```
##  License
This project is licensed under the [MIT License](LICENSE).
