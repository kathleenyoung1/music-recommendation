#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Attempting to run a model on the subset of test users
Usage:
    $ spark-submit model_v2.py <any arguments you wish to add>
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
    # comment
    """
    #######################
    # get the parquet files
    #######################
    print("get the parquet files")
    ms_train = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_train.parquet")
    ms_train = ms_train.repartition("user_id")
    #ms_train = ms_train.repartition(500, "user_id")
    ms_val = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_validation.parquet")
    #ms_val = ms_val.repartition(500, "user_id")
    ms_test = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet")
    #ms_test = ms_test.repartition(500, "user_id")

    ##########################################################
    # Sanity check: test has 100000 users, val has 10000 users
    # They do not overlap
    ##########################################################
    print("Sanity check: test has 100000 users, val has 10000 users")
    # create views
    ms_train.createOrReplaceTempView("ms_train")
    ms_val.createOrReplaceTempView("ms_val")
    ms_test.createOrReplaceTempView("ms_test")

    """
    """
    # check unique user count, val data
    print("Ensure test users and validation users are the same")
    print("count distinct users in validation")
    val_user_count = spark.sql("SELECT COUNT(DISTINCT(user_id)) FROM ms_val")
    val_user_count.show()

    # check unique user count, test data
    print("count distinct users in test")
    test_user_count = spark.sql("SELECT COUNT(DISTINCT(user_id)) FROM ms_test")
    test_user_count.show()

    # join user and test on user_id to check how many users overlap
    print("count distinct users in joined table")
    join_count = spark.sql("SELECT * FROM ms_val JOIN ms_test ON ms_val.user_id = ms_test.user_id")
    join_count.show()
    """
    """
    # comment
    # try filtering here
    val_and_test = ms_test.union(ms_val)[['user_id', 'track_id']]
    #val_and_test = val_and_test.repartition(500, "nuser_id")
    val_and_test.createOrReplaceTempView("val_and_test")
    #filter_train = spark.sql("SELECT ms_train.nuser_id, ms_train.count, ms_train.ntrack_id, ms_train.user_id, ms_train.track_id FROM val_and_test LEFT JOIN ms_train ON val_and_test.user_id = ms_train.user_id")
    filter_train = spark.sql("SELECT ms_train.count, ms_train.user_id, ms_train.track_id FROM val_and_test LEFT JOIN ms_train ON val_and_test.user_id = ms_train.user_id")
    filter_train.createOrReplaceTempView("filter_train")

    del ms_train
    spark.catalog.dropTempView("ms_train")

    #filter_train.show(10)
    filter_train.write.parquet("final-project-the-team/filter_train_int_ids.parquet")
    """
    """
    #####################
    # apply StringIndexer
    #####################
    # try reading the data?
    filter_train = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/filter_train_int_ids.parquet")
    # StringIndexer (on train first; then join with val, test)
    indexer = StringIndexer(inputCol="user_id", outputCol="nuser_id")#, inputCols="track_id", outputCols="ntrack_id")
    indexed = indexer.fit(filter_train).transform(filter_train)
    #indexed.show()

    indexer_track = StringIndexer(inputCol="track_id", outputCol="ntrack_id")
    filter_train = indexer_track.fit(indexed).transform(indexed)

    filter_train = filter_train.withColumn("nuser_id", filter_train["nuser_id"].cast(IntegerType()))
    filter_train = filter_train.withColumn("ntrack_id", filter_train["ntrack_id"].cast(IntegerType()))
    #filter_train = filter_train[['nuser_id', 'ntrack_id', 'count']]
    #print("filter_train after StringIndexer")
    #filter_train.show()

    # create a tempview of filter_train
    #ms_train = ms_train.repartition(500, "nuser_id")
    filter_train.createOrReplaceTempView("filter_train")
    #filter_train.show(10)
    # check new train count
    #print("Check unique user count for train with StringIndexer")
    #count_check = spark.sql("SELECT COUNT(DISTINCT(nuser_id)) FROM ms_train")
    #count_check.show()
    #print(ms_train.select("nuser_id").distinct().count())
    """
    filter_train = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/filter_train_actual_int_ids.parquet")
    filter_train.createOrReplaceTempView("filter_train")
    #print("Check unique user count for train with StringIndexer")
    #count_check = spark.sql("SELECT COUNT(DISTINCT(nuser_id)) FROM filter_train")
    #count_check.show()
    #filter_train.show(10)
    """
    #####################
    # apply StringIndexer
    #####################
    # StringIndexer (on train first; then join with val, test)
    indexer = StringIndexer(inputCol="user_id", outputCol="nuser_id")#, inputCols="track_id", outputCols="ntrack_id")
    indexed = indexer.fit(ms_train).transform(ms_train)
    #indexed.show()

    indexer_track = StringIndexer(inputCol="track_id", outputCol="ntrack_id")
    ms_train = indexer_track.fit(indexed).transform(indexed)

    ms_train = ms_train.withColumn("nuser_id", ms_train["nuser_id"].cast(IntegerType()))
    ms_train = ms_train.withColumn("ntrack_id", ms_train["ntrack_id"].cast(IntegerType()))

    #print("filter_train after StringIndexer")
    #filter_train.show()

    # create a tempview of filter_train
    #ms_train = ms_train.repartition(500, "nuser_id")
    ms_train.createOrReplaceTempView("ms_train")

    # check new train count
    #print("Check unique user count for train with StringIndexer")
    #count_check = spark.sql("SELECT COUNT(DISTINCT(nuser_id)) FROM ms_train")
    #count_check.show()
    #print(ms_train.select("nuser_id").distinct().count())
    """
    # try filtering here
    #val_and_test = ms_test.union(ms_val)[['user_id', 'track_id']]
    #val_and_test = val_and_test.repartition(500, "nuser_id")
    #val_and_test.createOrReplaceTempView("val_and_test")
    #filter_train = spark.sql("SELECT ms_train.nuser_id, ms_train.count, ms_train.ntrack_id, ms_train.user_id, ms_train.track_id FROM val_and_test LEFT JOIN ms_train ON val_and_test.user_id = ms_train.user_id")
    #filter_train = spark.sql("SELECT ms_train.count, ms_train.user_id, ms_train.track_id FROM val_and_test LEFT JOIN ms_train ON val_and_test.user_id = ms_train.user_id")
    #filter_train.createOrReplaceTempView("filter_train")

    #del ms_train
    #spark.catalog.dropTempView("ms_train")

    #filter_train.show(10)

    # comment
    """
    user_map = spark.sql("SELECT filter_train.nuser_id, filter_train.user_id FROM val_and_test LEFT JOIN filter_train ON val_and_test.user_id = filter_train.user_id")
    track_map = spark.sql("SELECT filter_train.ntrack_id, filter_train.track_id FROM val_and_test LEFT JOIN filter_train ON val_and_test.track_id = filter_train.track_id")
    #filter_train = ms_test.join(ms_train, ms_test.user_id == ms_train.user_id, 'left_outer').select(ms_train.user_id, ms_train['count'], ms_train.track_id)
    #filter_train.show()

    # create a dataframe with just the id columns to see if that helps
    #user_map = ms_train[["user_id", "nuser_id"]]
    #user_map = user_map.repartition(500, "user_id")
    user_map.createOrReplaceTempView("user_map")
    #track_map = ms_train[["track_id", "ntrack_id"]]
    #track_map = track_map.repartition(500, "track_id")
    track_map.createOrReplaceTempView("track_map")

    #######################################
    # Get test and val data with same index
    #######################################
    print("Get test and val data with same index")
    # join val set with filtered training set to get int ids
    #print("ms_val before join")
    #ms_val.show()

    ms_val = spark.sql("SELECT user_map.nuser_id, ms_val.count, ms_val.track_id  FROM ms_val JOIN user_map ON ms_val.user_id = user_map.user_id")
    #ms_val = ms_val.join(filter_train, ms_val.user_id == filter_train.user_id).select(filter_train.nuser_id, ms_val.count, ms_val.track_id)
    #ms_val = ms_val.repartition(500, "track_id")
    ms_val.createOrReplaceTempView("ms_val")
    #print("ms_val after user join")
    #ms_val.show()

    ms_val = spark.sql("SELECT track_map.ntrack_id, ms_val.count, ms_val.nuser_id FROM ms_val JOIN track_map ON ms_val.track_id = track_map.track_id")
    #ms_val = ms_val.join(filter_train, ms_val.track_id == filter_train.track_id).select(filter_train.ntrack_id, ms_val.count, ms_val.nuser_id)
    #ms_val = ms_val.repartition(500, "nuser_id")
    ms_val.createOrReplaceTempView("ms_val")
    #print("ms_val after track join")
    #ms_val.show()

    # check new count
    #print("Check unique user count for val with StringIndexer")
    #print(ms_val.select("nuser_id").distinct().count())


    # join test set with filtered training set to get int ids
    ms_test = spark.sql("SELECT user_map.nuser_id, ms_test.count, ms_test.track_id FROM ms_test JOIN user_map ON ms_test.user_id = user_map.user_id")
    #ms_val = ms_val.join(filter_train, ms_val.user_id == filter_train.user_id).select(filter_train.nuser_id, ms_val.count, ms_val.track_id)
    #ms_test = ms_test.repartition(500, "track_id")
    ms_test.createOrReplaceTempView("ms_test")
    #print("ms_val after user join")
    #ms_val.show()

    ms_test = spark.sql("SELECT track_map.ntrack_id, ms_test.count, ms_test.nuser_id FROM ms_test JOIN track_map ON ms_test.track_id = track_map.track_id")
    #ms_val = ms_val.join(filter_train, ms_val.track_id == filter_train.track_id).select(filter_train.ntrack_id, ms_val.count, ms_val.nuser_id)
    #ms_test = ms_test.repartition(500, "nuser_id")
    ms_test.createOrReplaceTempView("ms_test")
    #print("ms_val after track join")
    #ms_val.show()

    # check new count
    #print("Check unique user count for test with StringIndexer")
    #print(ms_test.select("nuser_id").distinct().count())

    # filter train
    print("Filter the training data on users in the testing set, apply StringIndexer")
    print("Filter training data")
    
    #val_and_test = ms_test.union(ms_val)[['nuser_id', 'ntrack_id']]
    #val_and_test = val_and_test.repartition(500, "nuser_id")
    #val_and_test.createOrReplaceTempView("val_and_test")
    #filter_train = spark.sql("SELECT ms_train.nuser_id, ms_train.count, ms_train.ntrack_id FROM val_and_test LEFT JOIN ms_train ON val_and_test.nuser_id = ms_train.nuser_id")
    #filter_train = ms_test.join(ms_train, ms_test.user_id == ms_train.user_id, 'left_outer').select(ms_train.user_id, ms_train['count'], ms_train.track_id)
    #filter_train.show()
    
    #filter_train = filter_train[['nuser_id', 'ntrack_id', 'count']]
    #ms_val = ms_val[['nuser_id', 'ntrack_id', 'count']]
    #ms_test = ms_test[['nuser_id', 'ntrack_id', 'count']]

    filter_train.printSchema()
    #filter_train.show()

    ms_val.printSchema()
    #ms_val.show()

    ms_test.printSchema()
    #ms_test.show()

    """
    # save as parquet files
    #filter_train.write.parquet("final-project-the-team/filter_train_actual_int_ids_all_cols.parquet")
    #ms_val.write.parquet("home/ky2132/final-project-the-team/val_int_ids.parquet")
    #ms_test.write.parquet("home/ky2132/final-project-the-team/test_int_ids.parquet")

    # trying to delete old variables out of memory to see if that helps...it did not
    #del ms_train
    #del ms_test
    #del indexer
    #del indexed
    #del indexer_track
    #del user_map
    #del track_map
    #del val_and_test
    #spark.catalog.dropTempView("ms_train")
    #spark.catalog.dropTempView("ms_val")
    #spark.catalog.dropTempView("ms_test")
    #spark.catalog.dropTempView("user_map")
    #spark.catalog.dropTempView("track_map")
    #spark.catalog.dropTempView("val_and_test")
    
    
    # train the model (just to make sure it works)
    start = time.time()
    als = ALS(rank=10,maxIter=10, regParam=0.1, numUserBlocks = 100, numItemBlocks = 100, userCol="nuser_id", itemCol="ntrack_id", ratingCol="count", coldStartStrategy="drop")
    print("fitting...")
    #filter_train = filter_train.repartition(500, "nuser_id")
    model = als.fit(filter_train)
    #print("saving...")
    #model.save("hdfs:/user/ky2132/final-project-the-team/model")
    end = time.time()
    print("it took ", end-start)
    # then come back to train hyperparameters
    """
    # evaluate the model (just to make sure it works)
    predictions = model.transform(ms_val)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))
    # then come back for different evaluation criteria

    # consider trying with more data
    """
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
