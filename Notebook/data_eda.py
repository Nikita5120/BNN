import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/churn.csv')
# Basic info
print(df.info())
print(df.describe())

# Missing values
print(df.isnull().sum())

# Plot churn count
sns.countplot(data=df, x='Churn')
plt.title("Churn Distribution")
plt.show()

# Save cleaned file if needed
df.dropna(inplace=True)
df.to_csv('data/churn_clean.csv', index=False)
