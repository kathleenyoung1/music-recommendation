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
    ms_val = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_validation.parquet")
    ms_small = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet")
    #ms_small = spark.read.parquet("hdfs:/user/bm106/pub/MSD").repartition('user_id')
    #ms_small = spark.read.parquet("hdfs:/user/drh382/final-project-the-team/ms_small.parquet")
    #ms_small = spark.read.parquet("hdfs:/user/ky2132/home/ky2132/final-project-the-team/ms_small.parquet")
    #ms_small = spark.read.parquet("hdfs:/user/ky2132/home/ky2132/final-project-the-team/ms_int_ids.parquet")
    #ms_small.printSchema()
    '''    
    ms_small.createOrReplaceTempView('ms_small')
    # print count unique ids so we can compare to the mapped dataset later
    user_query = spark.sql("SELECT COUNT(DISTINCT(user_id)) FROM ms_small")
    print("Count distinct user ids in the original dataset")
    user_query.show()
    track_query = spark.sql("SELECT COUNT(DISTINCT(track_id)) FROM ms_small")
    print("Count distinct track ids in the original dataset")
    track_query.show()
    '''
    
    # stringindex test
    indexer = StringIndexer(inputCol="user_id", outputCol="nuser_id")#, inputCols="track_id", outputCols="ntrack_id")
    indexed = indexer.fit(ms_small).transform(ms_small)
    indexed.show()

    indexer_track = StringIndexer(inputCol="track_id", outputCol="ntrack_id")
    new_df = indexer_track.fit(indexed).transform(indexed)
    new_df.show()

    new_df = new_df.withColumn("nuser_id", new_df["nuser_id"].cast(IntegerType()))
    new_df = new_df.withColumn("ntrack_id", new_df["ntrack_id"].cast(IntegerType()))

    new_df.printSchema()
    new_df.show()

    # stringindex val
    indexer = StringIndexer(inputCol="user_id", outputCol="nuser_id")#, inputCols="track_id", outputCols="ntrack_id")
    indexed = indexer.fit(ms_val).transform(ms_val)
    indexed.show()

    indexer_track = StringIndexer(inputCol="track_id", outputCol="ntrack_id")
    val_df = indexer_track.fit(indexed).transform(indexed)
    val_df.show()

    val_df = val_df.withColumn("nuser_id", val_df["nuser_id"].cast(IntegerType()))
    val_df = val_df.withColumn("ntrack_id", val_df["ntrack_id"].cast(IntegerType()))

    val_df.printSchema()
    val_df.show()
    '''
    print("Count distinct user ids after mapping")
    print(new_df.select("nuser_id").distinct().count())
    print("Count distinct user ids after mapping")
    print(new_df.select("ntrack_id").distinct().count())
    '''
    '''
    # fit model
    start = time.time()
    new_df = new_df[['nuser_id', 'ntrack_id', 'count']]
    
    als = ALS(maxIter=5, regParam=0.01, userCol="nuser_id", itemCol="ntrack_id", ratingCol="count", coldStartStrategy="drop")
    new_df.printSchema()
    new_df.show()
    # save to parquet so we don't have to keep running this
    #new_df.write.parquet("home/ky2132/final-project-the-team/ms_testdata_int_ids.parquet")
    print("fitting...")
    model = als.fit(new_df)
    # make predictions on val data (which is all we have for now)
    predictions = model.transform(val_df)

    # evaluate
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))
    #print("saving...")
    #model.save("hdfs:/user/ky2132/final-project-the-team/model")
    #model.save("hdfs:/user/ky2132/final-project-the-team/testdata_model")
    end = time.time()
    print("it took ", end-start)
    '''
    ranks = [10, 50]
    regParams = [0.1, 0.01]
    maxIters = 5

    for i in ranks:
        for j in regParams:
            startnow = time.time()
            als = ALS(maxIter=5, regParam=j,rank = i, userCol="nuser_id", itemCol="ntrack_id", ratingCol="count", coldStartStrategy="drop")
            endnow = time.time()
            #print("rank used: ", ranks, "regParams used: ", regParams, "maxIters used: ", maxIters, "it took ", endnow - startnow)
            this_model = als.fit(new_df)
            prediction = this_model.transform(val_df)
                
                
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                predictionCol="prediction") 
            rmse = evaluator.evaluate(prediction)        
            print("rank used: ", i, "regParams used: ", j, "maxIters used: ", maxIters, "rmse: ", rmse, "it took ", endnow - startnow) 


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
