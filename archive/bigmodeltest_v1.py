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


    ms_small = spark.read.parquet("hdfs:/user/bm106/pub/MSD").repartition('user_id')
    #ms_small = spark.read.parquet("hdfs:/user/drh382/final-project-the-team/ms_small.parquet")
    #ms_small = spark.read.parquet("hdfs:/user/ky2132/home/ky2132/final-project-the-team/ms_small.parquet")
    #ms_small.printSchema()
        
    ms_small.createOrReplaceTempView('ms_small')
    # print count unique ids so we can compare to the mapped dataset later
    user_query = spark.sql("SELECT COUNT(DISTINCT(user_id)) FROM ms_small")
    print("Count distinct user ids in the original dataset")
    user_query.show()
    track_query = spark.sql("SELECT COUNT(DISTINCT(track_id)) FROM ms_small")
    print("Count distinct track ids in the original dataset")
    track_query.show()

    

    indexer = StringIndexer(inputCol="user_id", outputCol="nuser_id")#, inputCols="track_id", outputCols="ntrack_id")
    indexed = indexer.fit(ms_small).transform(ms_small)
    indexed.show()

    indexer_track = StringIndexer(inputCol="track_id", outputCol="ntrack_id")
    new_df = indexer_track.fit(indexed).transform(indexed)
    #new_df.show()

    new_df = new_df.withColumn("nuser_id", new_df["nuser_id"].cast(IntegerType()))
    new_df = new_df.withColumn("ntrack_id", new_df["ntrack_id"].cast(IntegerType()))

    new_df.printSchema()
    new_df.show()

 
    # fit model
    start = time.time()
    als = ALS(maxIter=5, regParam=0.01, userCol="nuser_id", itemCol="ntrack_id", ratingCol="count", coldStartStrategy="drop")
    model = als.fit(new_df)
    model.save("hdfs:/user/drh382/final-project-the-team/model")
    end = time.time()
    print("it took ", end-start)

'''
    medtime = time.time()
    # make predictions on training data (which is all we have for now)
    predictions = model.transform(new_df)
    endtime = time.time()
    print("Time took to fit the model: ", medtime - start)
    print("Time took to fit and predict: ", endtime - start)

    # evaluate
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

'''


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
