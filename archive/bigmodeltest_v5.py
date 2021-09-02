#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Finally a model that works
Usage:
    $ spark-submit model_v5.py <any arguments you wish to add>
'''


# Import command line arguments and helper functions(if necessary)
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics

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
    # get the data
    #ms_train = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/03_filter_train_track_mapped.parquet")
    #ms_test = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/05_ms_test_track_mapped.parquet")
    ms_val = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/07_ms_val_user_mapped.parquet")

    # train the model
    #print("training...")
    #start = time.time()
    #als = ALS(rank=10,maxIter=10, regParam=0.1, numUserBlocks = 100, numItemBlocks = 100, userCol="nuser_id", itemCol="ntrack_id", ratingCol="count", coldStartStrategy="drop")
    #model = als.fit(ms_train)

    #print("saving...")
    #model.write().overwrite().save("hdfs:/user/ky2132/final-project-the-team/model_test")
    #end = time.time()
    #print("it took ", end-start)

    # get the model
    model = ALSModel.load("hdfs:/user/ky2132/final-project-the-team/model_test")
    #recs = model.recommendForAllUsers(numItems = 5)
    #recs.show(20, False)

    # generate recommendations for validation set
    recs = model.transform(ms_val)

    # prep for mean average precision
    users = recs.select('nuser_id').drop_duplicates()
    #users.show()
    ob_preds = recs.orderBy('prediction', ascending = False)
    #ob_preds.show()
    ob_labels = recs.orderBy('count', ascending = False)
    #ob_labels.show()
    predandlabels = []
    for user in users:
        pred = ob_preds.filter(ob_preds.nuser_id == user).select('ntrack_id').rdd.flatMap(lambda x: x).collect()[:500] # [:10] indicates number of recommendations. increase from 10 to 500
        #print(pred)
        label = ob_labels.filter(ob_labels.nuser_id == user).select('ntrack_id').rdd.flatMap(lambda x: x).collect()[:500] # # [:10] indicates number of recommendations. increase from 10 to 500
        #print(label)
        predandlabel = (pred, label)
        predandlabels.append(predandlabel)

    # get map
    metrics = RankingMetrics(sparkContext.parallelize(predandlabels))
    mavgpre = metrics.meanAveragePrecision
    print(f"map = {mavgpre}")

    # evaluate the model on test (not updated)
    #predictions = model.transform(ms_test)
    #evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
    #                            predictionCol="prediction")
    #rmse = evaluator.evaluate(predictions)
    #print("Root-mean-square error on test data = " + str(rmse))




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()
    sparkContext=spark.sparkContext

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
