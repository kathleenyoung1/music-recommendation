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

    raw_train = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_train.parquet").repartition('user_id')
    #raw_test = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet")
    #ms_small = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet")
    #ms_small = spark.read.parquet("hdfs:/user/drh382/final-project-the-team/ms_small.parquet")
    #ms_small = spark.read.parquet("hdfs:/user/ky2132/home/ky2132/final-project-the-team/ms_small.parquet")
    #ms_small.printSchema()

    raw_train.createOrReplaceTempView('raw_train')
    #raw_test.createOrReplaceTempView('raw_test')
 
    #create new train data where users have more than 20 song history

    user_query = spark.sql("SELECT a.user_id, raw_train.count, raw_train.track_id from (SELECT user_id, COUNT(DISTINCT(track_id)) FROM raw_train GROUP BY user_id having COUNT(DISTINCT(track_id))>20)a LEFT JOIN raw_train ON a.user_id = raw_train.user_id ")
    user_query.show()

    user_query.write.parquet("home/drh382/final-project-the-team/cf_over20train.parquet")



'''
    user_query = spark.sql("SELECT COUNT(user_id) from (SELECT user_id, COUNT(DISTINCT(track_id)) FROM raw_train GROUP BY user_id having COUNT(DISTINCT(track_id))>30)")
    user_query.show()

    user_query = spark.sql("SELECT count(user_id) from raw_train")
    user_query.show()



    user_query = spark.sql("SELECT COUNT(DISTINCT(user_id)) FROM raw_train")
    print("Count distinct user ids in the original dataset")
    user_query.show()
    track_query = spark.sql("SELECT COUNT(DISTINCT(track_id)) FROM raw_train")
    print("Count distinct track ids in the original dataset")
    track_query.show()

    user_query = spark.sql("SELECT COUNT(DISTINCT(user_id)) FROM raw_test")
    print("Count distinct user ids in the original dataset")
    user_query.show()
    track_query = spark.sql("SELECT COUNT(DISTINCT(track_id)) FROM raw_test")
    print("Count distinct track ids in the original dataset")
    track_query.show()
'''

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)


