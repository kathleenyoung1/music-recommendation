#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Python script to test literally just something in spark.
Usage:
    $ spark-submit test.py <file_path>
'''

# Import command line arguments and helper functions
import sys
#import bench

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# and statistics to automatically compute median
import statistics

# NOTE: the main function no longer requires a file_path argument from the
# command line. Instead, it loops through a list of file_paths which already
# hard-coded. This was more convenient than running the script
# 3 times each.

#def test_query(spark, file_path):

    # read the parquet file
#    msd = spark.read.parquet(file_path)

    # create a view to run SQL queries
#    msd.createOrReplaceTempView('msd')

    # test query
    #test = spark.sql('SELECT COUNT(*) FROM msd')
#    test = spark.sql('SELECT * FROM msd LIMIT 10')
#    test.show()

    # return uncomputed dataframe
#    return test



def main(spark, file_path):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    which_dataset : string, size of dataset to be analyzed
    '''
    # read the parquet file
    msd = spark.read.parquet(file_path)

    # create a view to run SQL queries
    msd.createOrReplaceTempView('msd')

    # test query
    test = spark.sql('SELECT COUNT(*) FROM msd')
    test = spark.sql('SELECT * FROM msd LIMIT 10')
    test.show()

    # return uncomputed dataframe
    #return test

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('project').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
