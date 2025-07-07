from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# # Simulation Churn Data
# Generate a synthetic dataset for binary classification
X,y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, n_classes=2, weights=[0.9, 0.1], random_state=42)


# ---- Random Forest with Stratified K-Fold Cross-Validation ----
# Create an object for Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5)
accuracies = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train a Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print("Stratified K-Fold Cross-Validation Accuracies: ", accuracies)



# ---- Random Forest without Stratifid K-Fold Cross-Validation ----
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ", accuracy)

