# this will unzip and copy the .csv file into your HDFS home directory
unzip resources/creditcard.csz.zip
hdfs dfs -copyFromLocal resources/creditcard.csv creditcard.csv