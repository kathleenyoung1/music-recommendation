#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Fourth step of parquet file creation
Usage:
    $ spark-submit 04_create_id_maps.py <any arguments you wish to add>
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
    filter_train = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/02_filter_train_for_map.parquet")

    # create user_map
    user_map = filter_train[["user_id", "nuser_id"]].drop_duplicates()

    # create track_map
    track_map = filter_train[["track_id", "ntrack_id"]].drop_duplicates()

    # write parquet files
    user_map.write.parquet("hdfs:/user/ky2132/final-project-the-team/04_user_map.parquet")
    track_map.write.parquet("hdfs:/user/ky2132/final-project-the-team/04_track_map.parquet")

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
