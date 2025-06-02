import pandas as pd
df = pd.read_csv('heart.csv')
print(df.head())

from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(16,10))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'])
plt.title("Decision Tree")
plt.show()

from sklearn.metrics import classification_report, accuracy_score

y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

dt_small = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_small.fit(X_train, y_train)
print("Smaller Tree Accuracy:", accuracy_score(y_test, dt_small.predict(X_test)))

import seaborn as sns

feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind='barh', figsize=(10,6))
plt.title("Feature Importance (Random Forest)")
plt.show()

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf, X, y, cv=5)
print("Cross-validation Accuracy:", cv_scores.mean())
