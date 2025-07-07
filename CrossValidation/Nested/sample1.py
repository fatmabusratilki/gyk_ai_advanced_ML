from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np

X,y = make_classification(n_samples=500, n_features=12, n_informative=10, n_classes=2, weights=[0.8,0.2], random_state=42)

model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100], # Number of trees in the forest
    'max_depth': [5, 10, None] # Maximum depth of the tree
}

# inside loop : hiperparameter tuning

inner_cv = StratifiedKFold(n_splits=3)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv)

# outer loop : model evaluation
outer_cv = StratifiedKFold(n_splits=5)
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)

print(f'Nested CV Accuracy: {nested_scores}')
print(f'Nested CV Accuracy: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}')
