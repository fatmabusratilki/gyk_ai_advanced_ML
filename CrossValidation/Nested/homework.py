import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
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

model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [10, 20, 50, 100], # Number of trees in the forest
    'max_depth': [3, 5, 10, None] # Maximum depth of the tree
}

# inside loop : hiperparameter tuning

inner_cv = StratifiedKFold(n_splits=3)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv)

# outer loop : model evaluation
outer_cv = StratifiedKFold(n_splits=5)
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)

print(f'Nested CV Accuracy: {nested_scores}')
print(f'Nested CV Accuracy: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}')