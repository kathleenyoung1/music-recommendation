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
    #ms_train = ms_train.repartition("nuser_id")
    #ms_test = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/05_ms_test_track_mapped.parquet")
    ms_val = spark.read.parquet("hdfs:/user/ky2132/final-project-the-team/07_ms_val_user_mapped.parquet")

    # train the model
    #print("training...")
    #start = time.time()

    alphas = [5]

    for j in alphas:
        #startnow = time.time()
        print(f"alpha = {j}")
        #print(f"maxIter = {k}")
        print("fitting...")
        #als = ALS(maxIter=20, regParam=0.001, rank = 100, alpha = j, numUserBlocks = 1000, numItemBlocks = 1000, userCol="nuser_id", itemCol="ntrack_id", ratingCol="count", coldStartStrategy="drop")
        als = ALS(maxIter=20, regParam=0.001, rank = 100, alpha = j, userCol="nuser_id", itemCol="ntrack_id", ratingCol="count", coldStartStrategy="drop")
        #endnow = time.time()

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
        print("evaluating...")
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

        recs = recs.sort(f.desc('prediction'))
        recs_2 = recs.filter(recs['count'] >= 5).sort(f.desc('count'))
        recs_3 = recs.sort(f.desc('count'))
        val_truerank=recs.select('nuser_id', f.collect_list('ntrack_id').over(w).alias('true_rank')).sort('nuser_id').distinct()
        val_truerank_2=recs_2.select('nuser_id', f.collect_list('ntrack_id').over(w).alias('true_rank')).sort('nuser_id').distinct()
        val_truerank_3=recs_3.select('nuser_id', f.collect_list('ntrack_id').over(w).alias('true_rank')).sort('nuser_id').distinct()

        scoreAndLabels = val_recrank.join(val_truerank,on=['nuser_id'],how='inner')
        scoreAndLabels_2 = val_recrank.join(val_truerank_2,on=['nuser_id'],how='inner')
        scoreAndLabels_3 = val_recrank.join(val_truerank_3,on=['nuser_id'],how='inner')


        rankLists_1=scoreAndLabels.select("rec_rank", "true_rank").rdd.map(lambda x: tuple([x[0],x[1]])).collect()
        rankLists_2=scoreAndLabels_2.select("rec_rank", "true_rank").rdd.map(lambda x: tuple([x[0],x[1]])).collect()
        rankLists_3=scoreAndLabels_3.select("rec_rank", "true_rank").rdd.map(lambda x: tuple([x[0],x[1][:500]])).collect()

        ranks_1 = spark.sparkContext.parallelize(rankLists_1)
        ranks_2 = spark.sparkContext.parallelize(rankLists_2)
        ranks_3 = spark.sparkContext.parallelize(rankLists_3)

        print("alpha used: ", j, "rmse: ", rmse)
        metrics = RankingMetrics(ranks_1)
        metrics_2 = RankingMetrics(ranks_2)
        metrics_3 = RankingMetrics(ranks_3)

        mavgpre_1 = metrics.meanAveragePrecision
        mavgpre_2 = metrics_2.meanAveragePrecision
        mavgpre_3 = metrics_3.meanAveragePrecision

        print(f"map_1 = {mavgpre_1}")
        print(f"map_2 = {mavgpre_2}")
        print(f"map_3 = {mavgpre_3}")

        preck_1 = metrics.precisionAt(500)
        preck_2 = metrics_2.precisionAt(500)
        preck_3 = metrics_3.precisionAt(500)

        print(f"prec_1 at k = {preck_1}")
        print(f"prec_2 at k = {preck_2}")
        print(f"prec_3 at k = {preck_3}")

        ndcgk_1 = metrics.ndcgAt(500)
        ndcgk_2 = metrics_2.ndcgAt(500)
        ndcgk_3 = metrics_3.ndcgAt(500)

        print(f"ndcg_1 at k = {ndcgk_1}")
        print(f"ndcg_2 at k = {ndcgk_2}")
        print(f"ndcg_2 at k = {ndcgk_3}")
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
