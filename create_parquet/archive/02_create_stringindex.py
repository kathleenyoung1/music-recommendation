#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Second step of parquet file creation
Usage:
    $ spark-submit 02_create_stringindex.py <any arguments you wish to add>
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
    #####################
    # apply StringIndexer
    #####################
    # read the data
    filter_train = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/01_filter_train.parquet")

    # StringIndexer
    indexer = StringIndexer(inputCol="user_id", outputCol="nuser_id")#, inputCols="track_id", outputCols="ntrack_id")
    indexed = indexer.fit(filter_train).transform(filter_train)

    indexer_track = StringIndexer(inputCol="track_id", outputCol="ntrack_id")
    filter_train = indexer_track.fit(indexed).transform(indexed)

    filter_train = filter_train.withColumn("nuser_id", filter_train["nuser_id"].cast(IntegerType()))
    filter_train = filter_train.withColumn("ntrack_id", filter_train["ntrack_id"].cast(IntegerType()))

    # write parquet file
    filter_train.write.parquet("hdfs:/user/ky2132/final-project-the-team/02_filter_train_for_map.parquet")

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
