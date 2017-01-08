package com.koala.ch06

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by seawalker on 2016/11/19.
  */
object MoviesFP {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val Array(wh, input, output) = args
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName).set("spark.sql.warehouse.dir", wh)
    val sc = new SparkContext(conf)

    val transactions = sc.textFile(input).map(_.split("\t")).map(_(1).split(","))

    val minSupport = 0.005
    val minConfidence = 0.1

    val fpg = new FPGrowth().setMinSupport(minSupport).setNumPartitions(10)

    val model = fpg.run(transactions)

    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
    }

    model.generateAssociationRules(minConfidence).collect().foreach { rule =>
      println(
        rule.antecedent.mkString("[", ",", "]")
          + " => " + rule.consequent .mkString("[", ",", "]")
          + ", " + rule.confidence)
    }

    model.save(sc, output)

    sc.stop()
  }
}
