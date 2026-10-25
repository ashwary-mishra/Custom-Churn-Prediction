import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Generate Synthetic Customer Data
np.random.seed(42)
n_samples = 1000

tenure = np.random.randint(1, 72, size=n_samples)
monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
total_charges = tenure * monthly_charges
support_tickets = np.random.poisson(lam=1.5, size=n_samples)
contract_type = np.random.choice([0, 1, 2], size=n_samples) # 0: Month-to-Month, 1: One Year, 2: Two Year

churn_prob = 1 / (1 + np.exp(-(-1.5 - 0.04*tenure + 0.01*monthly_charges + 0.5*support_tickets + 0.8*(contract_type==0))))
churn = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    'Tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'SupportTickets': support_tickets,
    'ContractType': contract_type,
    'Churn': churn
})

# 2. Features & Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# 5. Save Model
with open('churn_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("\n--------------------------------------------------")
print("SUCCESS: Model train ho gaya aur 'churn_rf_model.pkl' save ho gayi!")
print("--------------------------------------------------\n")
