package com.koala.ch05

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by seawalker on 2016/11/23.
  */
object Regressions {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val Array(wh, mode, input) = args

    val spark = SparkSession.builder
      .config("spark.sql.warehouse.dir", wh)
      .master(mode)
      .appName(this.getClass.getName)
      .getOrCreate()

    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("libsvm").load(input)
    data.persist

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    trainingData.persist
    testData.persist

    lr(data)
    glr(data)

    spark.stop()
  }


  def lr(training: DataFrame) = {
    val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    // Fit the model
    val lrModel = lr.fit(training)
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
    val predictions = lrModel.transform(training)
    predictions.select("prediction", "label", "features").show(100)
  }

  def glr(training: DataFrame) = {
    val glr = new GeneralizedLinearRegression().setFamily("gaussian").setLink("identity").setMaxIter(10).setRegParam(0.3)
    // Fit the model
    val model = glr.fit(training)
    // Print the coefficients and intercept for generalized linear regression model
    println(s"Coefficients: ${model.coefficients}")
    println(s"Intercept: ${model.intercept}")
    // Summarize the model over the training set and print out some metrics
    val summary = model.summary
    println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
    println(s"T Values: ${summary.tValues.mkString(",")}")
    println(s"P Values: ${summary.pValues.mkString(",")}")
    println(s"Dispersion: ${summary.dispersion}")
    println(s"Null Deviance: ${summary.nullDeviance}")
    println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
    println(s"Deviance: ${summary.deviance}")
    println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
    println(s"AIC: ${summary.aic}")
    println("Deviance Residuals: ")
    summary.residuals().show()

    val predictions = model.transform(training)
    predictions.select("prediction", "label", "features").show(100)
  }


}