### Importing Data

from pyspark.sql import SparkSession
import pandas as pd
import pickle

# Create the Spark context.
spark = SparkSession.builder.appName("cc_fraud").getOrCreate()

# Read the csv from your HDFS home directory
cc_data = spark.read.csv(
    "creditcard.csv", header=True, mode="DROPMALFORMED",inferSchema=True
)

# Pull the Spark dataframe to a local Pandas dataframe.
cc_dataframe = cc_data.toPandas()

# Show the columns in the data.
cc_dataframe[1:10].transpose()

