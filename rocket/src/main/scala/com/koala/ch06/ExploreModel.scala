package com.koala.ch06

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.fpm.FPGrowthModel
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by seawalker on 2016/11/19.
  */

object ExploreModel {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val Array(wh, modelPath, moviePath) = args
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName).setMaster("local[4]").set("spark.sql.warehouse.dir", wh)
    val sc = new SparkContext(conf)
    val model = FPGrowthModel.load(sc, modelPath)

    val userCnt = 222329

    val movies = sc.textFile(moviePath).map(_.split(",")).map{
      case terms =>  (terms(0), "%s_%s".format(terms(1), terms(2)))
    }.collect().toMap

    // 2-item sets
    val supports = model.freqItemsets.filter(_.items.length <= 2).collect()
      .map(itemSet => (itemSet.items.map(_.toString.toInt).sorted.mkString(","), itemSet.freq * 1.0D / userCnt )).toMap

    val moviesBC = sc.broadcast(movies)
    val supportsBC = sc.broadcast(supports)

    val ruleViews = model.generateAssociationRules(0.2).filter(_.antecedent.length == 1)
      .map {
        rule =>
          val left = rule.antecedent.map(_.toString.toInt).sorted.mkString(",")
          val right =  rule.consequent.mkString(",")
          val confidence = rule.confidence
          val leftS = supportsBC.value.get(left).get
          val rightS = supportsBC.value.get(right).get
          val support = supportsBC.value.get(rule.antecedent.union(rule.consequent).map(_.toString.toInt).sorted.mkString(",")).get
          (left, right, Seq(support, confidence, support / leftS / rightS))
    }.map {
      case (left, right, indicators) =>
        (moviesBC.value.get(left).get, moviesBC.value.get(right).get, indicators.map(x => "%.4f".format(x)))
    }

    ruleViews.take(100).foreach(println)

    sc.stop()
  }
}