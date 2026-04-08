import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# -------------------- LOAD DATA --------------------
diabetes_dataset = pd.read_csv('dataset/diabetes.csv')

print(diabetes_dataset.head())
print("Shape:", diabetes_dataset.shape)
print(diabetes_dataset.describe())

print(diabetes_dataset['Outcome'].value_counts())
print(diabetes_dataset.groupby('Outcome').mean())

# -------------------- SPLIT DATA --------------------
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X.head())
print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print(X.shape, X_train.shape, X_test.shape)

# -------------------- TRAIN MODEL --------------------
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X_train, Y_train)

# -------------------- ACCURACY --------------------
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Training Accuracy:', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Test Accuracy:', test_data_accuracy)

# -------------------- SAMPLE PREDICTION --------------------
input_data = (5,166,72,19,175,25.8,0.587,51)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
proba = classifier.predict_proba(input_data_reshaped)

print("Prediction:", prediction)
print("Confidence:", np.max(proba)*100)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')

# -------------------- SAVE MODEL --------------------
filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# -------------------- LOAD MODEL --------------------
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

prediction = loaded_model.predict(input_data_reshaped)
proba = loaded_model.predict_proba(input_data_reshaped)

print("Loaded Model Prediction:", prediction)
print("Loaded Model Confidence:", np.max(proba)*100)

# -------------------- FEATURE NAMES --------------------
print("\nFeature Columns:")
for column in X.columns:
    print(column)