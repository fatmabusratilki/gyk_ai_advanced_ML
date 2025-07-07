import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed()

df = pd.DataFrame({
    "product_price" : np.random.uniform(10,100,2000), #1
    "employee_experience_years" : np.random.randint(1,20,2000), #4
    "customer_order_count" : np.random.randint(1,50,2000), #5
    "shipper_delivery_time_days" : np.random.randint(1,10,2000), #2
    "order_total_amount" : np.random.uniform(100,1000,2000) #3
})

X = df.drop("order_total_amount",axis=1)
y = df["order_total_amount"]

#L1 (Lasso)

lasso = LassoCV(cv=7)
lasso.fit(X,y)

#PCA 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#PCA3
print("Açıklanan varyans oranları : ",pca.explained_variance_ratio_)
print("Toplam açıklanan ",pca.explained_variance_ratio_.sum())


#Görselleştirme PCA4

loadings = pd.DataFrame(pca.components_.T,columns=["PCA1","PCA2"],index=X.columns)
print("PCA bileşen yükleri :\n",loadings)


feature_importance = pd.Series(lasso.coef_,index=X.columns)
print("Özellik önemleri :\n",feature_importance)

#DR diğer yöntemler nelerdir?
#Görselleştirme yap