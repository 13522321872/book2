package com.koala.ch07.rdd

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by seawalker on 2016/11/21.
  */
object MoviesALSRDDBased {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val Array(wh, mode, ratingsPath, output) = args
    val conf = new SparkConf()
      .setMaster(mode)
      .setAppName(this.getClass.getSimpleName)

    val sc = new SparkContext(conf)
    // load ratings
    val ratings = sc.textFile(ratingsPath).map { line =>
      val fields = line.split(",")
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }

    val numRatings = ratings.count
    val numUsers = ratings.map(_.user).distinct.count
    val numMovies = ratings.map(_.product).distinct.count

    println(s"ratings: $numRatings, users: $numUsers, movies: $numMovies.")

    val Array(training, validation, test) = ratings.randomSplit(Array(0.6, 0.2, 0.2)).map(_.persist)

    val numTraining = training.count
    val numValidation = validation.count
    val numTest = test.count
    println(s"training: $numTraining, validation: $numValidation, test: $numTest.")

    val ranks = List(8, 10, 12, 16)
    val lambdas = List(0.01, 0.1, 1.0)
    val numIters = List(6, 8, 10)

    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      val model = ALS.train(training, rank, numIter, lambda)
      val validationRmse = computeRmse(model, validation)

      println(s"$rank, $lambda, $numIter's RMSE (validation): $validationRmse")

      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    val testRmse = computeRmse(bestModel.get, test)
    println(s"The best model was trained with ($bestRank,$bestLambda,$bestLambda), and its RMSE(test) is $testRmse ")
    val baselineRmse = computeBaseline(training.union(validation), test, numTest)
    val improvement = (baselineRmse - testRmse) / baselineRmse * 100
    println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

    bestModel.get.save(sc, output)
    sc.stop();
  }

  def computeBaseline( data: RDD[Rating], test: RDD[Rating], numTest:Long) : Double ={
    val meanRating = data.map(_.rating).mean
    val baselineRmse = math.sqrt(test.map(x => (meanRating - x.rating) * (meanRating - x.rating))
      .reduce(_ + _) / numTest)
    baselineRmse
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating]) : Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map{ x =>
      ((x.user, x.product), x.rating)
    }.join(data.map(x => ((x.user, x.product), x.rating))).values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }

}