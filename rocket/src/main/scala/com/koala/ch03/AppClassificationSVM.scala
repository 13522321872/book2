package com.koala.ch03

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by seawalker on 2016/11/17.
  */
object AppClassificationSVM {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    /**
      * input: libsvm file.
      * output: model store path.
      * mode: yarn-client, yarn-cluster or local[*].
      */
    val Array(input, output, mode) = args
    val sc = new SparkContext(new SparkConf().setAppName(this.getClass.getSimpleName).setMaster(mode))

    /* 5 * 4 / 2 = 10 */
    //TODO 0 or 1, 0 or 2, ...
    val data = MLUtils.loadLibSVMFile(sc, input).filter(_.label < 2.0).map{
      case lp =>
        val label = if (lp.label >= 1.0) 1 else 0
        new LabeledPoint(label, lp.features)
    }



    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val svmAlg = new SVMWithSGD()
    svmAlg.optimizer
      .setNumIterations(200)
      .setRegParam(0.01)
      //.setUpdater(new L1Updater)

    val model = svmAlg.run(training)
    // Clear the default threshold.
    model.clearThreshold()
    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println("Area under ROC = " + auROC)
    // Save and load model
    // model.save(sc, "target/tmp/scalaSVMWithSGDModel")
    // val sameModel = SVMModel.load(sc, "target/tmp/scalaSVMWithSGDModel")
  }

}
