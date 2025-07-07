import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Blending

X,y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

X_train,X_test,y_train,y_test =  train_test_split(X,y, test_size=0.2, random_state=42)

base_learners = [
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("svc", SVC(probability=True, random_state=42))
]

meta_learner = LogisticRegression()

stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)


stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)

print(accuracy_score(y_test,y_pred))