#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Hyperparameter search for alpha
Usage:
    $ spark-submit alpha_search.py
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
from pyspark.sql import functions as f
from pyspark.sql import Window

def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object

    '''
    # fix checkpointing issue
    spark.sparkContext.setCheckpointDir('hdfs:/user/ky2132/final-project-the-team/temp')

    # get the data
    ms_train = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/03_filter_train_track_mapped.parquet")
    ms_val = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/07_ms_val_user_mapped.parquet")

    # hyperparameters to search
    maxIters = [30, 50, 100]

    for k in maxIters:
        startnow = time.time()
        als = ALS(maxIter=k, regParam=0.001,rank = 100, alpha = 1, numUserBlocks = 1000, numItemBlocks = 1000, userCol="nuser_id", itemCol="ntrack_id", ratingCol="count", coldStartStrategy="drop")
        model = als.fit(ms_train)
        endnow = time.time()

        # generate recommendations for validation set
        recs = model.transform(ms_val)

        # get rmse
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
        rmse = evaluator.evaluate(recs)

        # prep for ranking metrics
        # get unique users
        users = recs.select('nuser_id').drop_duplicates()

        # get top 500 recommendations from model for each user
        val_rec = model.recommendForUserSubset(users,500)
        val_rec = val_rec.select('nuser_id','recommendations',f.posexplode('recommendations')).drop('recommendations')
        val_rec = val_rec.select('nuser_id',f.expr('col.ntrack_id'),f.expr('col.rating'))
        w = Window.partitionBy('nuser_id')
        val_recrank = val_rec.select('nuser_id',f.collect_list('ntrack_id').over(w).alias('rec_rank')).sort('nuser_id').distinct()

        # get the relevant set (listen count >= 5)
        recs = recs.filter(recs['count'] >= 5).sort(f.desc('count'))
        val_truerank=recs.select('nuser_id', f.collect_list('ntrack_id').over(w).alias('true_rank')).sort('nuser_id').distinct()

        # convert to rdd required by RankingMetrics
        scoreAndLabels = val_recrank.join(val_truerank,on=['nuser_id'],how='inner')
        rankLists=scoreAndLabels.select("rec_rank", "true_rank").rdd.map(lambda x: tuple([x[0],x[1]])).collect()
        ranks = spark.sparkContext.parallelize(rankLists)

        # evalute
        print("maxIters used: ", k, "rmse: ", rmse, "it took ", endnow - startnow)
        metrics = RankingMetrics(ranks)
        mavgpre = metrics.meanAveragePrecision
        print(f"map = {mavgpre}")
        preck = metrics.precisionAt(500)
        print(f"prec at k = {preck}")
        ndcgk = metrics.ndcgAt(500)
        print(f"ndcg at k = {ndcgk}")

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()
    sparkContext=spark.sparkContext

    main(spark)
