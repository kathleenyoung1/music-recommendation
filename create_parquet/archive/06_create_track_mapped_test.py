#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Fourth step of parquet file creation
Usage:
    $ spark-submit 05_create_mapped_test.py <any arguments you wish to add>
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
    ms_test = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/05_ms_test_user_mapped.parquet")
    #ms_test = ms_test.sort("track_id")
    #ms_test = ms_test.repartition("track_id")
    #track_map = track_map.sort("track_id")
    #track_map = track_map.repartition("track_id")
    ms_test.createOrReplaceTempView("ms_test")

    # test if there are overlapping track ids between filter_train, test
    track_map = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/04_track_map.parquet")
    track_map.createOrReplaceTempView("track_map")
    #ms_test = spark.sql("SELECT track_map.ntrack_id, ms_test.count, ms_test.nuser_id FROM ms_test LEFT JOIN track_map ON ms_test.track_id = track_map.track_id")
    overlap = spark.sql("SELECT COUNT(DISTINCT(ms_test.track_id)) FROM ms_test JOIN track_map ON ms_test.track_id = track_map.track_id")
    #overlap = spark.sql("SELECT COUNT(*) FROM ms_test")
    overlap.show()
    overlap = spark.sql("SELECT COUNT(DISTINCT(track_id)) FROM ms_test")
    overlap.show()
    #ms_test.show(10)
    """
    # StringIndexer
    indexer_track = StringIndexer(inputCol="track_id", outputCol="ntrack_id")
    ms_test = indexer_track.fit(ms_test).transform(ms_test)
    ms_test = ms_test.withColumn("ntrack_id", ms_test["ntrack_id"].cast(IntegerType()))

    # write parquet file
    ms_test.write.parquet("hdfs:/user/ky2132/final-project-the-team/06_FINAL_ms_test.parquet")
    #ms_test.repartition(100)
    #ms_test.write.mode('overwrite').parquet("hdfs:/user/ky2132/final-project-the-team/06_final_ms_test.parquet")
    """

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
