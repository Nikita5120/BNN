import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.bnn import BayesianNN

# Load data
df = pd.read_csv('data/churn_clean.csv')

# Drop customerID
df = df.drop(columns=["customerID"])

# Encode target
if df['Churn'].dtype == object:
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].values

# One-hot encode categorical columns
X = pd.get_dummies(X)

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Define model
model = BayesianNN(input_dim=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss()

# Train model
for epoch in range(50):
    model.train()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.4f}")

# Evaluate model
model.eval()
with torch.no_grad():
    predictions = torch.stack([model(X_test) for _ in range(100)])
    mean_preds = predictions.mean(0).numpy()
    print("ROC AUC:", roc_auc_score(y_test, mean_preds))

# ✅ Save model + metadata for compatibility with Streamlit
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': X_train.shape[1]
}, 'app/bnn_model.pth')
torch.save(X.columns.tolist(), 'app/columns.pth')
