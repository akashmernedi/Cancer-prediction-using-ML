# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import jaccard_score
import pickle

# Loading data into a dataframe
df = pd.read_csv("cell_samples.csv")
print(df.head())

# Length of the dataframe
print(len(df))

# Checking data types of the dataframe
print(df.info())

# Dropping the rows that are not numerical in BareNuc
df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc'] = df['BareNuc'].astype('int')
print(df.dtypes)

# Extracting the features from main dataframe
feature_df = df[['ID', 'Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
print(X[0:5])

df['Class'] = df['Class'].astype('int')
y = np.asarray(df['Class'])
print(y [0:5])

# Train-Test split the dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Fitting the model
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train) 

# Predicting the values
yhat = clf.predict(X_test)
yhat [0:5]

print("Jaccard Score: %.2f" % jaccard_score(y_test, yhat, pos_label=2))

# Dumping the model into pickle
with open('svc_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
 
# Loading the saved data from pickle   
with open('svc_model.pkl', 'rb') as f:
    stored_clf = pickle.load(f)
    
print(stored_clf)



    