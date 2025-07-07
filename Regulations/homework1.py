
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# PostgreSQL bağlantı URL'i
database_url = "postgresql+psycopg2://postgres:test@localhost:5432/northwind2"
engine = create_engine(database_url, echo=True)

# SQL sorgusu
query = """
SELECT 
    od.unit_price, od.quantity, od.discount,
    (od.unit_price * od.quantity * (1 - od.discount)) AS total_amount,
    p.category_id AS product_category,
    p.supplier_id AS supplier,
    c.country AS customer_country,
    e.city AS employee_city,
    EXTRACT(MONTH FROM o.order_date) AS order_month,
    s.company_name AS shipper_name
FROM order_details od
JOIN orders o ON od.order_id = o.order_id
JOIN products p ON p.product_id = od.product_id
JOIN customers c ON o.customer_id = c.customer_id
JOIN employees e ON o.employee_id = e.employee_id
JOIN shippers s ON o.ship_via = s.shipper_id
"""

df = pd.read_sql(query, engine)

# Hedef değişken
y = df["total_amount"]

# Bağımsız değişkenler
X = df.drop(columns=["total_amount", "unit_price", "quantity", "discount"])

# Kategorik değişkenleri one-hot encode
X_encoded = pd.get_dummies(X, drop_first=True)

# Eğitim ve test setleri
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Lasso Regression
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)

# Sonuçlar
print("Lasso Regression MSE:", mse_lasso)
print("Linear Regression MSE:", mse_linear)

# Lasso katsayıları
lasso_coeffs = pd.Series(lasso.coef_, index=X_encoded.columns)
print("\nLasso Katsayıları (önemli olanlar):")
print(lasso_coeffs[lasso_coeffs != 0].sort_values(ascending=False))
