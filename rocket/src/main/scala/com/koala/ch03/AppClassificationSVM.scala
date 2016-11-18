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
    val data = MLUtils.loadLibSVMFile(sc, input).cache()
    val labels = data.map(_.label).distinct().collect().sorted.combinations(2).map(x => (x.mkString("_"), x))

    labels.foreach {
      case (tag, tuple) =>
        val parts = data.filter(lp => tuple.contains(lp.label)).map{
          case lp =>
            val label = if (lp.label == tuple(0)) 0 else 1
            new LabeledPoint(label, lp.features)
        }
        val splits = parts.randomSplit(Array(0.6, 0.4), seed = 11L)
        val training = splits(0).cache()
        val test = splits(1)
        val svmAlg = new SVMWithSGD()
        svmAlg.optimizer
          .setNumIterations(200)
          .setRegParam(0.01)
        val model = svmAlg.run(training)
        // Clear the default threshold.
        model.clearThreshold()
        val scoreAndLabels = test.map { point =>
          val score = model.predict(point.features)
          (score, point.label)
        }

        // Get evaluation metrics.
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        val auc = metrics.areaUnderROC()
        model.save(sc, output + tag)
    }
    sc.stop()
  }

}
