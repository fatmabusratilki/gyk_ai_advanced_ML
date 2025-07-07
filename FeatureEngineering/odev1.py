import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

database_url = "postgresql+psycopg2://postgres:BT1234@localhost:5432/GYKNorthwind"
engine = create_engine(database_url, echo=True)

df = pd.read_sql("""
    select p.product_name, o.ship_name, od.unit_price, od.quantity
    from orders o
    join order_details od on o.order_id = od.order_id
    join products p on p.product_id = od.product_id 
""",engine)

df["total"] = df["unit_price"] * df["quantity"]

df["product_shipper_cross"] = df["product_name"] + "_" + df["ship_name"]

df_encoded = pd.get_dummies(df[["product_shipper_cross"]], prefix="cross") 
df_model = pd.concat([df, df_encoded], axis=1)


# Task 1 : Kodu Northwind veritabanına taşıyın.
# Task 2 : XgBoost ve Lineer Regression ile de karşılaştırın.

X = df_encoded
y = df_model["total"]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

# Linear Regression
modelLinear = LinearRegression()
modelLinear.fit(X_train, y_train)

y_prediction = modelLinear.predict(X_test)
rmseLinear = np.sqrt(mean_squared_error(y_test, y_prediction))

print("Linear Regression RMSE :", rmseLinear)

# XGBoost Regression
modelXGBoost = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
modelXGBoost.fit(X_train, y_train)

y_prediction = modelXGBoost.predict(X_test)
rmseXGBoost = np.sqrt(mean_squared_error(y_test, y_prediction))

print("XGBoost Regression RMSE :", rmseXGBoost)
