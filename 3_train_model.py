### Model Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import pickle
from pyspark.sql import SparkSession

# Load the data
spark = SparkSession.builder.appName("cc_fraud").getOrCreate()

# Read the csv from your HDFS home directory
spark_cc_data = spark.read.csv(
    "creditcard.csv", header=True, mode="DROPMALFORMED",inferSchema=True
)

# Load the data
cc_data = spark_cc_data.toPandas()

# First split the data into features `X` and lables `y`. Then split the data further into 
# test and train data sets.

X = cc_data.iloc[:,1:len(cc_data.columns)-1]
y = cc_data.iloc[:,len(cc_data.columns)-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lets first try a Random Forest classifier.

param_numTrees = 20
param_maxDepth = 20 
param_impurity = 'gini' 

randF=RandomForestClassifier(n_jobs=10,
                             n_estimators=param_numTrees, 
                             max_depth=param_maxDepth, 
                             criterion = param_impurity,
                             random_state=0)

# The magical `model.fit()` :)
randF.fit(X_train, y_train)


# Test the model accuracy using AUROC

predictions_rand=randF.predict(X_test)
pd.crosstab(y_test, predictions_rand, rownames=['Actual'], colnames=['Prediction'])

auroc = roc_auc_score(y_test, predictions_rand)
ap = average_precision_score(y_test, predictions_rand)
print(auroc, ap)

# This model does quite well. A littel too well. I suspect its over fitting.


# Now lets try a Logistic Regression classifier

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=5, solver='liblinear', multi_class='ovr')
logreg.fit(X_train.values, y_train.values.reshape(-1, 1))

# Test the model accuracy using AUROC
predictions_log=logreg.predict(X_test.values)
pd.crosstab(y_test, predictions_log, rownames=['Actual'], colnames=['Prediction'])


auroc = roc_auc_score(y_test, predictions_log)
ap = average_precision_score(y_test, predictions_log)
print(auroc, ap)

# This model does not do quite as well.

## Save the model

pickle.dump(randF, open("cc_model.pkl","wb"))

spark.stop()





