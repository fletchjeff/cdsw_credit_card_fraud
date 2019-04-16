from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import cdsw

from pyspark.sql import SparkSession

# Load the data
spark = SparkSession.builder.appName("cc_fraud").getOrCreate()

# Read the csv from your HDFS home directory
spark_cc_data = spark.read.csv(
    "creditcard.csv", header=True, mode="DROPMALFORMED",inferSchema=True
)

# Load the data
cc_data = spark_cc_data.toPandas()

X = cc_data.iloc[:,1:len(cc_data.columns)-1]
y = cc_data.iloc[:,len(cc_data.columns)-1]

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42
)

param_numTrees = int(sys.argv[1])
param_maxDepth = int(sys.argv[2])
param_impurity = sys.argv[3]


randF=RandomForestClassifier(
  n_jobs=10,
  n_estimators=param_numTrees, 
  max_depth=param_maxDepth, 
  criterion = param_impurity,
  random_state=0
)

randF.fit(X_train, y_train)

predictions_rand=randF.predict(X_test)
auroc = roc_auc_score(y_test, predictions_rand)
ap = average_precision_score(y_test, predictions_rand)


cdsw.track_metric("auroc", round(auroc,2))
cdsw.track_metric("ap", round(ap,2))

pickle.dump(randF, open("cc_model_checl.pkl","wb"))

cdsw.track_file("cc_model_checl.pkl")