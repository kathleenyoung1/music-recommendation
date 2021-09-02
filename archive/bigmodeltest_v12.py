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
from pyspark.sql import functions as f
from pyspark.sql import Window

def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object

    '''
    # get the data
    ms_train = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/03_filter_train_track_mapped.parquet")
    ms_test = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/05_ms_test_track_mapped.parquet")
    ms_val = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/07_ms_val_user_mapped.parquet")

    # train the model
    #print("training...")
    start = time.time()

    ranks = [10, 50, 100]
    regParams = [1, 0.1, 0.01, 0.001]
    maxIters = [10, 20]

    for i in ranks:
        for j in regParams:
            for k in maxIters:
                startnow = time.time()
                #als = ALS(maxIter=k, regParam=j,rank = i, numUserBlocks = 100, numItemBlocks = 100, userCol="nuser_id", itemCol="ntrack_id", ratingCol="count", coldStartStrategy="drop")
                als = ALS(maxIter=k, regParam=j,rank = i, userCol="nuser_id", itemCol="ntrack_id", ratingCol="count", coldStartStrategy="drop")
                endnow = time.time()

    #als = ALS(rank=10,maxIter=10, regParam=0.1, numUserBlocks = 100, numItemBlocks = 100, userCol="nuser_id", itemCol="ntrack_id", ratingCol="count", coldStartStrategy="drop")
                model = als.fit(ms_train)

    #print(s_model = als.fit(new_df)

    #model.write().overwrite().save("hdfs:/user/ky2132/final-project-the-team/model_test")
    #end = time.time()
    #print("it took ", end-start)

    # get the model
    #model = ALSModel.load("hdfs:/user/drh382/final-project-the-team/model_test")
    #recs = model.recommendForAllUsers(numItems = 5)
    #recs.show(20, False)

    # generate recommendations for validation set
                recs = model.transform(ms_val)
    # get rmse
                evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
                rmse = evaluator.evaluate(recs)

    # prep for mean average precision
                users = recs.select('nuser_id').drop_duplicates()

                val_rec = model.recommendForUserSubset(users,500)
    #val_rec.show()

                val_rec = val_rec.select('nuser_id','recommendations',f.posexplode('recommendations')).drop('recommendations')
                val_rec = val_rec.select('nuser_id',f.expr('col.ntrack_id'),f.expr('col.rating'))

                w= Window.partitionBy('nuser_id')
                val_recrank=val_rec.select('nuser_id',f.collect_list('ntrack_id').over(w).alias('rec_rank')).sort('nuser_id').distinct()

                #recs = recs.sort(f.desc('prediction'))
                #recs = recs.filter(recs['count'] >= 5).sort(f.desc('count'))
                recs = recs.sort(f.desc('count'))
                val_truerank=recs.select('nuser_id', f.collect_list('ntrack_id').over(w).alias('true_rank')).sort('nuser_id').distinct()

                scoreAndLabels = val_recrank.join(val_truerank,on=['nuser_id'],how='inner')

                rankLists=scoreAndLabels.select("rec_rank", "true_rank").rdd.map(lambda x: tuple([x[0],x[1][:500]])).collect()
                ranks = spark.sparkContext.parallelize(rankLists)
                print("rank used: ", i, "regParams used: ", j, "maxIters used: ", k, "rmse: ", rmse, "it took ", endnow - startnow)
                metrics = RankingMetrics(ranks)
                mavgpre = metrics.meanAveragePrecision
                print(f"map = {mavgpre}")
                preck = metrics.precisionAt(500)
                print(f"prec at k = {preck}")
                ndcgk = metrics.ndcgAt(500)
                print(f"ndcg at k = {ndcgk}")
    '''


    #users.show()
    ob_preds = val_rec.orderBy('prediction', ascending = False)
    #ob_preds.show()
    ob_labels = val_rec.orderBy('count', ascending = False)
    #ob_labels.show()
    predandlabels = []
    for user in users:
        pred = ob_preds.filter(ob_preds.nuser_id == user).select('ntrack_id').rdd.flatMap(lambda x: x).collect() # [:10] indicates number of recommendations. increase from 10 to 500
        #print(pred)
        label = ob_labels.filter(ob_labels.nuser_id == user).select('ntrack_id').rdd.flatMap(lambda x: x).collect() # # [:10] indicates number of recommendations. increase from 10 to 500
        #print(label)
        predandlabel = (pred, label)
        predandlabels.append(predandlabel)
    #print(predandlabels)
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

    preck = metrics.precisionAt(500)
    print(f"prec at k = {preck}")
    ndcgk = metrics.ndcgAt(500)
    print(f"ndcg at k = {ndcgk}")
    #recallk = metrics.recallAt
    #print(recallk)

    '''
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final').config('spark.blacklist.enabled', False).getOrCreate()
    sparkContext=spark.sparkContext

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
