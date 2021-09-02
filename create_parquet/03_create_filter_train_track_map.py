#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Third step of parquet file creation
Usage:
    $ 03_create_stringindex_filter_train.py <any arguments you wish to add>
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

    # read the data
    filter_train = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/02_filter_train_user_mapped.parquet")
    filter_train = filter_train.repartition("track_id")
    track_map = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/00_track_map.parquet")
    track_map = track_map.repartition("track_id")

    filter_train.createOrReplaceTempView("filter_train")
    track_map.createOrReplaceTempView("track_map")

    # map user ids
    filter_train = spark.sql("SELECT track_map.ntrack_id, filter_train.count, filter_train.nuser_id FROM filter_train LEFT JOIN track_map ON filter_train.track_id = track_map.track_id")

    # write parquet
    filter_train.write.parquet("hdfs:/user/ky2132/final-project-the-team/03_filter_train_track_mapped.parquet")

    """
    # get only the columns we want
    filter_train = filter_train[['nuser_id', 'ntrack_id', 'count']]

    # write to parquet
    filter_train.write.parquet("hdfs:/user/ky2132/final-project-the-team/03_final_train.parquet")
    """

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
