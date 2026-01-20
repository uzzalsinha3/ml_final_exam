import pandas as pd
import numpy as np

#sklearn preprocessing

from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


#Regression model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  RandomForestClassifier

#metrices
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.metrics import classification_report, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# data loading
# Loading data
df = pd.read_csv("diabetes.csv")
#shape
print(df.shape)
# Displaying first 5 rows
print(df.head())
print("\n\nBasic Statistics:")
print(df.describe())



col_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replacing 0 with NaN for these columns
for col in col_with_zero:
    df[col] = df[col].replace(0, np.nan)
# checking null values
print(df.isnull().sum())
# Missing Value Imputation with median
for col in col_with_zero:
     df[col]=df[col].fillna(df[col].median())
# checking null values
print("After imputation, checking null values")
print(df.isnull().sum())

# Outlier Detection using IQR Method and capping with lower and upper bound values
def outlier_iqr_capping(df, columns):
    df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Capping
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df
# extracting features
features_col = df.drop('Outcome', axis=1)
# Capping the outliers
df = outlier_iqr_capping(df,features_col )

# Prepare features and target
X = df.drop('Outcome',axis=1)
y = df['Outcome']
#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,  random_state=42, stratify=y)
print("Training data size: ", X_train.shape[0], "rows")
print("Test data size: ", X_test.shape[0], "rows")
#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)


# Pipeline
num_features = X.columns.tolist()

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, num_features)
])


## Selected Model: Random Forest Classifier

# Justification:

# Binary classification problem

# Handle noise and outliers effectively

# Works well on non-linear data

# Prevents overfitting.

# model training
# from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)


# cross validation
from sklearn.model_selection import cross_val_score
# Accuracy
cv_scores_acc = cross_val_score(
    pipeline, X_train, y_train, cv=5, scoring='accuracy',n_jobs=-1
)

print("CV Mean Accuracy:", cv_scores_acc.mean())
print("CV Std Dev:", cv_scores_acc.std())
# Precision
cv_scores_preci = cross_val_score(
    pipeline, X_train, y_train, cv=5, scoring='precision', n_jobs=-1
)
print("CV Mean Precision:", cv_scores_preci.mean())
print("CV Std Dev:", cv_scores_preci.std())

# Recall
cv_scores_rec = cross_val_score(
    pipeline, X_train, y_train, cv=5, scoring='recall', n_jobs=-1
)
print("CV Mean Recall:", cv_scores_rec.mean())
print("CV Std Dev:", cv_scores_rec.std())

# F1
cv_scores_f1 = cross_val_score(
    pipeline, X_train, y_train, cv=5, scoring='f1', n_jobs=-1
)

print("CV Mean F1:", cv_scores_f1.mean())
print("CV Std Dev:", cv_scores_f1.std())

# ROC-AUC
cv_scores_roc = cross_val_score(
    pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
)
print("CV Mean ROC-AUC:", cv_scores_roc.mean())
print("CV Std Dev:", cv_scores_roc.std())


# Hyperparameter tuning
# Define parameter grid
param_grid = {
    'classifier__n_estimators': [10, 20, 50, 100],
    'classifier__max_depth': [3, 5, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__class_weight': ['balanced', None]
}


grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# best model selection
best_model = grid_search.best_estimator_
print(best_model)
best_classifier = best_model.named_steps['classifier']
print()
print(f" n_estimators: {best_classifier.n_estimators}")
print(f" max_depth: {best_classifier.max_depth}")
print(f" min_samples_split: {best_classifier.min_samples_split}")
print(f" min_samples_leaf: {best_classifier.min_samples_leaf}")
print(f" max_features: {best_classifier.max_features}")
print(f" class_weight: {best_classifier.class_weight}")
print(f" random_state: {best_classifier.random_state}")



# model evaluation

y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model

with open("Diabetes Prediction System.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Random Forest Classifier pipeline saved as Diabetes Prediction System.pkl")