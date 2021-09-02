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
    ms_val = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/06_ms_val_track_mapped.parquet")
    ms_val = ms_val.repartition("user_id")
    user_map = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/00_user_map.parquet")
    user_map = user_map.repartition("user_id")

    ms_val.createOrReplaceTempView("ms_val")
    user_map.createOrReplaceTempView("user_map")

    # map user ids
    ms_val = spark.sql("SELECT user_map.nuser_id, ms_val.count, ms_val.ntrack_id FROM ms_val LEFT JOIN user_map ON ms_val.user_id = user_map.user_id")

    # write parquet
    ms_val.write.parquet("hdfs:/user/ky2132/final-project-the-team/07_ms_val_user_mapped.parquet")

    """
    # get only the columns we want
    ms_val = ms_val[['nuser_id', 'ntrack_id', 'count']]

    # write to parquet
    ms_val.write.parquet("hdfs:/user/ky2132/final-project-the-team/03_final_train.parquet")
    """

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
