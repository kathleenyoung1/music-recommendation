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
    ms_train = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_train.parquet")
    ms_train = ms_train.repartition("track_id")
    ms_val = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_validation.parquet")
    ms_test = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet")

    all_data = ms_train.union(ms_test).union(ms_val)

    # StringIndexer
    #indexer = StringIndexer(inputCol="user_id", outputCol="nuser_id")#, inputCols="track_id", outputCols="ntrack_id")
    #ms_train = indexer.fit(ms_train).transform(ms_train)

    indexer_track = StringIndexer(inputCol="track_id", outputCol="ntrack_id")
    all_data = indexer_track.fit(all_data).transform(all_data)

    #ms_train = ms_train.withColumn("nuser_id", ms_train["nuser_id"].cast(IntegerType()))
    all_data = all_data.withColumn("ntrack_id", all_data["ntrack_id"].cast(IntegerType()))

    # create user_map
    #user_map = ms_train[["user_id", "nuser_id"]].drop_duplicates()

    # create track_map
    track_map = all_data[["track_id", "ntrack_id"]].drop_duplicates()

    # write parquet file
    track_map.repartition("ntrack_id")
    #user_map.write.parquet("hdfs:/user/ky2132/final-project-the-team/00_user_map.parquet")
    track_map.write.parquet("hdfs:/user/ky2132/final-project-the-team/00_track_map.parquet")

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
