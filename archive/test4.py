#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit lab_3_storage_template_code.py <any arguments you wish to add>
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

    #ms_small = spark.read.parquet("hdfs:/user/drh382/final-project-the-team/ms_small.parquet")
    ms_small = spark.read.parquet("hdfs:/user/ky2132/home/ky2132/final-project-the-team/ms_small.parquet")
    #ms_small.printSchema()

    ms_small.createOrReplaceTempView('ms_small')
    #query = spark.sql('SELECT * FROM ms_small LIMIT 100')
    #query.show()

# change ids from strings to integers
    int_user_id = ms_small.select('user_id').distinct().select('user_id', F.monotonically_increasing_id().alias('new_user_id'))
    int_track_id = ms_small.select('track_id').distinct().select('track_id', F.monotonically_increasing_id().alias('new_track_id'))

# get total unique users and songs
    unique_users = int_user_id.count()
    unique_songs = int_track_id.count()
    print('Number of unique users: {0}'.format(unique_users))
    print('Number of unique songs: {0}'.format(unique_songs))

# join dataframes
    df = ms_small.join(int_user_id, 'user_id').join(int_track_id, 'track_id')

#checking new df
    df.show(10)






# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
