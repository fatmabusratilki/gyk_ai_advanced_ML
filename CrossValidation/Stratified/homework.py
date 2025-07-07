import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

database_url = "postgresql+psycopg2://postgres:sifreeee@localhost:5432/Gyk1Northwind"  # chane here to your database URL northwind database

engine = create_engine(database_url, echo=True)

df = pd.read_sql("""
    SELECT
        c.customer_id,
        COUNT(o.order_id) as order_count,
        AVG(o.freight) as avg_freight,
        COUNT(DISTINCT o.ship_country) as unique_ship_countries,
        MAX(o.order_date) - MIN(o.order_date) as active_days,
        COUNT(DISTINCT EXTRACT(YEAR FROM o.order_date)) as order_years
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id
""", engine)

# Hedef değişken
df["label"] = df["order_count"].apply(lambda x: 1 if x > 1 else 0)

# Eksik değer kontrolü
df.fillna(0, inplace=True)

# X ve y
X = df.drop(columns=["customer_id", "label", "order_count"])
y = df["label"]

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5)
accuracies = []
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print("Stratified CV Accuracies:", accuracies)
print("Average Accuracy:", round(sum(accuracies)/len(accuracies), 4))

# ---- Results ----
# Stratified CV Accuracies: [0.9473684210526315, 1.0, 1.0, 1.0, 1.0]
# Average Accuracy: 0.9895