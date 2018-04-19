import pandas as pd

pd.options.mode.chained_assignment = None

df = pd.read_csv('student_records.csv')

feature_names = ['OverallGrade', 'Obedient', 'ResearchScore', 'ProjectScore']
training_features = df[feature_names]

outcome_name = ['Recommend']
outcome_labels = df[outcome_name]

print(outcome_labels)

numeric_feature_names = ['ResearchScore', 'ProjectScore']
categorical_feature_names = ['Obedient', 'OverallGrade']

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

ss.fit(training_features[numeric_feature_names])
training_features[numeric_feature_names] = ss.transform(training_features[numeric_feature_names])

# print(training_features)

# print(training_features[categorical_feature_names])

training_features = pd.get_dummies(training_features, columns=categorical_feature_names)

# print(training_features)

categorical_engineered_features = list(set(training_features.columns)-set(categorical_feature_names))

# print(categorical_engineered_features)

from sklearn.linear_model import LogisticRegression
import numpy as np

lr = LogisticRegression()
# print(outcome_labels['Recommend'])
# print(np.array(outcome_labels['Recommend']))
model = lr.fit(training_features, outcome_labels['Recommend'])

# print(model)

pred_labels = model.predict(training_features)
# print(pred_labels)
actual_labels = np.array(outcome_labels['Recommend'])
# print(actual_labels)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print('Accuracy: ', float(accuracy_score(actual_labels, pred_labels))*100, '%')
print(classification_report(actual_labels, pred_labels))

from sklearn.externals import joblib
import os

if not os.path.exists('Model'):
    os.mkdir('Model')
if not os.path.exists('Scalar'):
    os.mkdir('Scalar')

joblib.dump(model, r'Model/model.pickle')
joblib.dump(ss, r'Scalar/scalar.pickle')

