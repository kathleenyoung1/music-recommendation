#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
First step of parquet file creation
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
    # get the parquet files
    print("Getting parquet files")
    ms_train = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_train.parquet")
    ms_train = ms_train.repartition("user_id")
    ms_val = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_validation.parquet")
    ms_test = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet")

    ms_train.createOrReplaceTempView("ms_train")
    ms_val.createOrReplaceTempView("ms_val")
    ms_test.createOrReplaceTempView("ms_test")

    # filter train data on users in test, val
    # get users in val and test set
    val_and_test = ms_test.union(ms_val)[['user_id', 'track_id']]
    val_and_test.createOrReplaceTempView("val_and_test")

    # filter train
    filter_train = spark.sql("SELECT ms_train.user_id, ms_train.track_id, ms_train.count FROM val_and_test LEFT JOIN ms_train ON val_and_test.user_id = ms_train.user_id")
    filter_train.createOrReplaceTempView("filter_train")

    # delete ms_train variables (I don't know if this actuall makes a difference)
    del ms_train
    spark.catalog.dropTempView("ms_train")

    # write to parquet
    filter_train.write.parquet("hdfs:/user/ky2132/final-project-the-team/01_filter_train.parquet")

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
