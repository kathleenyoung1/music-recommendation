#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
First attempt at a model on 1% of the data
Usage:
    $ spark-submit model_v1.py <any arguments you wish to add>
'''


# Import command line arguments and helper functions(if necessary)
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

import os
from pyspark.sql.types import *
from pyspark.sql import functions as F

def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object

    '''
    #raw_train = spark.read.parquet("home/drh382/final-project-the-team/cf_filteredtrain.parquet")

    #raw_train = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_train.parquet").repartition('user_id')
    train_data = spark.read.parquet("home/drh382/final-project-the-team/cf_over20train.parquet")



    raw_val = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_validation.parquet")
    raw_test = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet")
    #ms_small = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet")
    #ms_small = spark.read.parquet("hdfs:/user/drh382/final-project-the-team/ms_small.parquet")
    #ms_small = spark.read.parquet("hdfs:/user/ky2132/home/ky2132/final-project-the-team/ms_small.parquet")
    #ms_small.printSchema()

    train_data.createOrReplaceTempView('train_data')
    raw_val.createOrReplaceTempView('raw_val')
    raw_test.createOrReplaceTempView('raw_test')

    
    user_query1 = spark.sql("Select user_id, count, track_id, 'train' AS name from train_data")
    user_query1.show() 
    train = user_query1
    user_query1.registerTempTable('dataset')

    user_query2 = spark.sql("Select user_id, count, track_id, 'val' AS name from raw_val")
    user_query2.show() 
    val = user_query2

    user_query3 = spark.sql("Select user_id, count, track_id, 'test' AS name from raw_test")
    user_query3.show() 
    test = user_query3


    train.union(test).union(val).registerTempTable("table_test")


    #test.show()

    #alldata = [train, val, test]
    user_query = spark.sql("select count(distinct(user_id)) from table_test")
    user_query.show()



#only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)


