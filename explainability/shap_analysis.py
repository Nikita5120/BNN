import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../data/churn_clean.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']

model = RandomForestClassifier().fit(X, y)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values[1], X)