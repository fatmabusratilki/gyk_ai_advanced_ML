# Dorğusal bağlantılı veri varsa ridge kullanın.
# Eğer doğrusal bağlantılı veri yoksa linear regression kullanın.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

np.random.seed(42)

n = 500

df = pd.DataFrame({
    "ad_budget": np.random.uniform(1000, 5000, n),
    "sales_team_size": np.random.randint(1000, 2000, n),
    "delivery_days": np.random.randint(1000, 1900, n),
})

# Multiple linear connected data, Multicollinearity
df["marketing_expense"] = df["ad_budget"] + np.random.normal(0,200,n)

df["sales"] = (df["ad_budget"] * 0.5 + df["sales_team_size"] * 0.3 + df["delivery_days"] * 2 + np.random.normal(0, 200, n))

X = df.drop("sales", axis=1)
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ridge = Ridge(alpha=100)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred2 = linear_model.predict(X_test)

lasso = Lasso(alpha=200)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)



print("Ridge MSE:", mean_squared_error(y_test, y_pred))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred2))
print("Lasso MSE:", mean_squared_error(y_test, y_pred_lasso))
print("Ridge Coefficients:\n", pd.Series(ridge.coef_, index=X.columns))
print("Linear Regression Coefficients:\n", pd.Series(linear_model.coef_, index=X.columns))
print("Lasso Coefficients:\n", pd.Series(lasso.coef_, index=X.columns))