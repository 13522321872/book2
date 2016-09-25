package com.koala.ch03

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

/**
  * Created by haixiang on 2016/9/24.
  */

object AppClassification {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .config("spark.sql.warehouse.dir", "file:///E:/git/spark-warehouse")
      .master("local[4]")
      .appName(this.getClass.getName)
      .getOrCreate()


    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load("E:/git/data/ch02/stage02")
    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
    // Train a NaiveBayes model.
    val model = new NaiveBayes().fit(trainingData)
    // Select example rows to display.
    val predictions = model.transform(testData)
    predictions.show()
    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy: " + accuracy)
    spark.stop()
  }
}
