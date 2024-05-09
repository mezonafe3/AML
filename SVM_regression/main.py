# imports
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             mean_squared_error, mean_absolute_error)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# dataset info

performance_ds = pd.read_csv('./datasets/student_performance.csv')
# diabetes_ds = load_diabetes()
# cancer_ds = load_breast_cancer()

# print('\nFeatures: ', insurance_ds.feature_names, '\n')  # information related to the patient.
#
# print('\nLabels: ', insurance_ds.target[:5], '\n')  # progression of diabetes (minimum is 25 and maximum is 346).
#
# print('\nShape: ', insurance_ds.data.shape, '\n')  # features shape (442 sample with 10 features each).

print('\nDescription: ', performance_ds.describe(), '\n')

# feature extraction

# applying one hot encoding with label encoder

# le = LabelEncoder()
# cat_columns = ['sex', 'smoker', 'region']
# for column in cat_columns:
#     insurance_ds[column] = le.fit_transform(insurance_ds[column])
# num_columns = ['age', 'bmi', 'children']

# applying one hot encoding with get dummies

performance_ds = pd.get_dummies(performance_ds)

print(performance_ds)

print('\nColumns names: ', performance_ds.columns, '\n')
print('\nColumns types: ', performance_ds.columns.dtype, '\n')

# splitting data

# X_train, X_test, y_train, y_test = train_test_split(insurance_ds[['age', 'sex', 'bmi', 'children', 'smoker', 'region']],
# insurance_ds['charges'], test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(performance_ds[['Hours Studied', 'Previous Scores', 'Sleep Hours',
                                                                    'Sample Question Papers Practiced',
                                                                    'Extracurricular Activities_No',
                                                                    'Extracurricular Activities_Yes']],
                                                    performance_ds['Performance Index'], test_size=0.3,
                                                    random_state=76)  # 70% train and 30% test

# scaling features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# reg = LinearRegression(n_jobs=-1)
# param_grid = {'kernel': ['linear', 'rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
# reg = RandomizedSearchCV(SVR(), param_grid, scoring='neg_mean_squared_error', cv=KFold(n_splits=5))
# clf = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=KFold(n_splits=5))


# generating model
# print('_________________________________')
# print('\nfitting data...\n')
# reg.fit(X_train_scaled, y_train)  # fitting the data into the classifier
# print('done...\n')
# print('_________________________________')
# print('\nsaving model...\n')
# joblib.dump(reg, 'svm_model.joblib')
# print('model saved successfully!')
# print('_________________________________')

reg = joblib.load('svm_model.joblib')
print('\npredicting...\n')
y_pred = reg.predict(X_test_scaled)  # predicting the target values
print('done...\n')
print('_________________________________')

pca_insurance = PCA(n_components=1)
X_train_PCA = pca_insurance.fit_transform(X_train)
X_test_PCA = pca_insurance.transform(X_test)

print('\nX_train shape: ', X_train.shape, '\nX_test shape: ', X_test.shape, '\ny_train shape: ', y_train.shape,
      '\ny_test shape: ', y_test.shape, '\ny_pred shape: ', y_pred.shape, '\nX_train_PCA shape: ', X_train_PCA.shape,
      '\nX_test_PCA shape: ', X_test_PCA.shape)

# plot the predicted values against the true values
plt.scatter(X_train_PCA, y_train, color='darkorange',
            label='data')
plt.plot(X_test_PCA, y_pred, color='cornflowerblue',
         label='prediction')
plt.legend()
plt.show()

# evaluating regression model
best_params = reg.best_params_
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
print('_________________________________')
print('\nMean squared error:', MSE, '\n')
print('\nMean absolute error: ', MAE, '\n')
print('_________________________________')
print('\nBest model parameters: ', best_params, '\n')
print('_________________________________')


# evaluating classification model

# accuracy = accuracy_score(y_test, y_pred)
# f1_score = f1_score(y_test, y_pred, average='macro')
# recall = recall_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
#
# print('\nModel accuracy: ', accuracy, '\n')
# print('\nF1 score: ', f1_score, '\n')
# print('\nRecall score: ', recall, '\n')
# print('\nPrecision score: ', precision, '\n')

def calculate_student_performance(hours_studied=int(input('\nplease enter hours studied: \n')),
                                  previous_scores=int(input('\nplease enter previous scores: \n')),
                                  sleep_hours=int(input('\nplease enter sleep hours: \n')),
                                  sample_question=int(input(
                                      '\nplease enter how many sample question papers you practiced: \n'))):
    while True:
        ea_string = input('\ndid you participate in any extracurricular activities? (Y/N)\n')
        if ea_string == 'Y' or ea_string == 'y':
            ea_yes = 1
            ea_no = 0
            break
        elif ea_string == 'N' or ea_string == 'n':
            ea_no = 1
            ea_yes = 0
            break
        else:
            print('please enter a valid response (Y/N)!')
    student_stats = [hours_studied, previous_scores, sleep_hours, sample_question, ea_no, ea_yes]
    student_stats = np.array(student_stats).reshape(1, -1)
    return student_stats


stats = calculate_student_performance()
stats_scaled = scaler.transform(stats)
prediction = reg.predict(stats_scaled)
print('\nYour expected score is : ', prediction)
