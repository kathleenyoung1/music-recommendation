#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Sanity checks
Usage:
    $ spark-submit 01_create_filter_train.py <any arguments you wish to add>
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

from pyspark.ml.feature import StringIndexer
import time
from pyspark.sql.types import IntegerType
import numpy as np

def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object

    '''
    # read files
    ms_val = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_validation.parquet")
    ms_test = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet")
    filter_train = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/01_filter_train.parquet")
    string_index_filter_train = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/03_filter_train_track_mapped.parquet")
    user_map = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/00_user_map.parquet")
    track_map = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/00_track_map.parquet")
    final_test = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/05_ms_test_track_mapped.parquet")
    final_val = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/07_ms_val_user_mapped.parquet")

    ms_val.createOrReplaceTempView("ms_val")
    ms_test.createOrReplaceTempView("ms_test")
    filter_train.createOrReplaceTempView("filter_train")
    string_index_filter_train.createOrReplaceTempView("string_index_filter_train")
    user_map.createOrReplaceTempView("user_map")
    track_map.createOrReplaceTempView("track_map")
    final_test.createOrReplaceTempView("final_test")
    final_val.createOrReplaceTempView("final_val")

    print("Rows ms_val")
    query = spark.sql("SELECT COUNT(*) FROM ms_val")
    query.show()

    print("Distinct users ms_val")
    query = spark.sql("SELECT COUNT(DISTINCT(user_id)) FROM ms_val")
    query.show()

    print("Distinct tracks ms_val")
    query = spark.sql("SELECT COUNT(DISTINCT(track_id)) FROM ms_val")
    query.show()

    print("Rows ms_test")
    query = spark.sql("SELECT COUNT(*) FROM ms_test")
    query.show()

    print("Distinct users ms_test")
    query = spark.sql("SELECT COUNT(DISTINCT(user_id)) FROM ms_test")
    query.show()

    print("Distinct tracks ms_test")
    query = spark.sql("SELECT COUNT(DISTINCT(track_id)) FROM ms_test")
    query.show()

    print("Rows filter train")
    query = spark.sql("SELECT COUNT(*) FROM filter_train")
    query.show()

    print("Distinct users filter_train")
    query = spark.sql("SELECT COUNT(DISTINCT(user_id)) FROM filter_train")
    query.show()

    print("Distinct tracks filter_train")
    query = spark.sql("SELECT COUNT(DISTINCT(track_id)) FROM filter_train")
    query.show()

    print("Rows filter train with stringindex")
    query = spark.sql("SELECT COUNT(*) FROM string_index_filter_train")
    query.show()

    print("Distinct users filter_train with stringindex")
    query = spark.sql("SELECT COUNT(DISTINCT(nuser_id)) FROM string_index_filter_train")
    query.show()

    print("Distinct tracks filter_train with stringindex")
    query = spark.sql("SELECT COUNT(DISTINCT(ntrack_id)) FROM string_index_filter_train")
    query.show()

    print("count(*) user map")
    query = spark.sql("SELECT COUNT(*) FROM user_map")
    query.show()

    print("count(*) track map")
    query = spark.sql("SELECT COUNT(*) FROM track_map")
    query.show()

    print("Rows final_val")
    query = spark.sql("SELECT COUNT(*) FROM ms_val")
    query.show()

    print("Distinct users final val")
    query = spark.sql("SELECT COUNT(DISTINCT(nuser_id)) FROM final_val")
    query.show()

    print("Distinct tracks final val")
    query = spark.sql("SELECT COUNT(DISTINCT(ntrack_id)) FROM final_val")
    query.show()

    print("Rows final_test")
    query = spark.sql("SELECT COUNT(*) FROM ms_test")
    query.show()

    print("Distinct users final test")
    query = spark.sql("SELECT COUNT(DISTINCT(nuser_id)) FROM final_test")
    query.show()

    print("Distinct tracks final test")
    query = spark.sql("SELECT COUNT(DISTINCT(ntrack_id)) FROM final_test")
    query.show()

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
