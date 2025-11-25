# iris_decision_tree.py (or a notebook cell)
"""
Task: Preprocess Iris dataset, train Decision Tree, evaluate accuracy/precision/recall.
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# 1. Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target  # already numeric (0,1,2)
feature_names = iris.feature_names
target_names = iris.target_names

# If you had missing values: example imputation (here not necessary)
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy='mean')
# X = imputer.fit_transform(X)

# 2. (If labels were strings) encode labels - iris already numeric
# encoder = LabelEncoder()
# y = encoder.fit_transform(y_raw)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Decision Tree
clf = DecisionTreeClassifier(random_state=42, max_depth=4)  # max_depth to reduce overfitting
clf.fit(X_train, y_train)

# 5. Predict & Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')

print("Accuracy:", acc)
print("Precision (macro):", prec_macro)
print("Recall (macro):", recall_macro)
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Optionally: save model (joblib)
# import joblib
# joblib.dump(clf, 'iris_decision_tree.joblib')
