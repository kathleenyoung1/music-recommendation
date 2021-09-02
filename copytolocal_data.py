#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
First attempt at a model on 1% of the data
Usage:
    $ spark-submit model_v1.py <any arguments you wish to add>
'''

#ip install lightfm

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
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

#import pyarrow
#import fastparquet
'''
from lightfm.data import Dataset

from lenskit.algorithms.basic import Bias
from lenskit.metrics.predict import rmse
from lenskit.batch import predict
from lenskit.algorithms.als import BiasedMF
'''
import pandas as pd

def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    ''' 
    train = spark.read.parquet("home/drh382/final-project-the-team/cf_filteredtrain.parquet").repartition(1)
    #val = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_validation.parquet")
    #ms_small = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet")
    #df = pd.read_parquet("home/drh382/final-project-the-team/cf_filteredtrain.parquet")
    #aa = df.repartition(1).to_csv('ms_train1.csv')
    #train.write.csv("home/drh382/final-project-the-team/ms_train.csv")

    train.createOrReplaceTempView('train')
    train.printSchema()
    #train.show()
    #query = spark.sql("SELECT count(distinct(user_id)) from train")
    #query.show()

    train.write.csv("home/drh382/final-project-the-team/csv/ms_train.csv")
    '''





    ms_train = spark.read.parquet("hdfs:/user/drh382/final-project-the-team/03_filter_train_track_mapped.parquet").repartition(1)
    ms_val = spark.read.parquet("hdfs:/user/drh382/final-project-the-team/05_ms_test_track_mapped.parquet").repartition(1)
    ms_test = spark.read.parquet("hdfs:/user/drh382/final-project-the-team/07_ms_val_user_mapped.parquet").repartition(1)

    ms_train.createOrReplaceTempView('ms_train')
    ms_val.createOrReplaceTempView('ms_val')
    ms_test.createOrReplaceTempView('ms_test')

    ms_train.write.csv("home/drh382/final-project-the-team/data/ms_train.csv")
    ms_val.write.csv("home/drh382/final-project-the-team/data/ms_val.csv")
    ms_test.write.csv("home/drh382/final-project-the-team/data/ms_test.csv")









# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
