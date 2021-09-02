#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
First attempt at a model on 1% of the data
Usage:
    $ spark-submit model_v1.py <any arguments you wish to add>
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
    
    # print count unique ids so we can compare to the mapped dataset later
    user_query = spark.sql("SELECT COUNT(DISTINCT(user_id)) FROM ms_small")
    print("Count distinct user ids in the original dataset")
    user_query.show()
    track_query = spark.sql("SELECT COUNT(DISTINCT(track_id)) FROM ms_small")
    print("Count distinct track ids in the original dataset")
    track_query.show()

    # unfortunately monotonically_increasing_id() converts to 64 bit integers,
    # and we need 32 bit integers. Here is a new solution:

    # select distinct user, track ids
    int_user_id = ms_small.select('user_id').distinct()
    int_track_id = ms_small.select('track_id').distinct()

    # convert to rdd, zip each id with its index
    int_user_id = int_user_id.rdd.zipWithIndex()
    int_track_id = int_track_id.rdd.zipWithIndex()

    # unfortunately zipWithIndex stores each id in a list of one?
    # so we have to get them back out of the lists
    # note that the rdds don't have column names, just indices
    int_user_id = int_user_id.map(lambda x: (x[0][0], x[1])) # 0 = user_id, 1 = new_user_id
    int_track_id = int_track_id.map(lambda x: (x[0][0], x[1])) # 0 = track_id, 1 = new_track_id

    # convert back to df
    int_user_id = int_user_id.toDF()
    int_track_id = int_track_id.toDF()

    # fix the column names
    int_user_id = int_user_id.withColumnRenamed("_1", "user_id").withColumnRenamed("_2", "new_user_id")
    int_track_id = int_track_id.withColumnRenamed("_1", "track_id").withColumnRenamed("_2", "new_track_id")

    # show
    #int_user_id.show(10)
    #int_track_id.show(10)

    # test if it worked
    print("Count distinct user ids after mapping")
    print(int_user_id.select("new_user_id").distinct().count())
    print("Count distinct user ids after mapping")
    print(int_track_id.select("new_track_id").distinct().count())

    # join dataframes
    df = ms_small.join(int_user_id, 'user_id').join(int_track_id, 'track_id')

    # checking new df
    df.show(10)

    # running the model code from the documentation: https://spark.apache.org/docs/2.4.7/ml-collaborative-filtering.html
    # get the columns we need
    df = df[['new_user_id','new_track_id','count']]

    # fit model
    als = ALS(maxIter=5, regParam=0.01, userCol="new_user_id", itemCol="new_track_id", ratingCol="count", coldStartStrategy="drop")
    model = als.fit(df)

    # make predictions on training data (which is all we have for now)
    predictions = model.transform(df)

    # evaluate
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
