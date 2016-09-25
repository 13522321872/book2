package com.koala.ch03

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by haixiang on 2016/9/24.
  */

object AppTrainingData {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val sc = new SparkContext(new SparkConf().setMaster("local[1]").setAppName(this.getClass.getName))

    val rdd = sc.textFile("E:/git/data/ch02/stage01")
      .map(_.split("~", -1))
      .filter(!_(3).isEmpty)
      .map(x => (x(2), x(3).split(":")))

    val indexs = rdd.flatMap(_._2).distinct().collect()

    val training = rdd.map{
      case (label, terms) =>
        val svm = terms.map(v => (v, 1)).groupBy(_._1).map {
          case (v, vs) => (v, vs.length)
        }.map{
          case (v, cnt) => (indexs.indexOf(v) + 1, cnt)
        }.toSeq
          .sortBy(_._1)
          .map(x => "" + x._1 + ":" + x._2)
          .mkString(" ")

        (label, svm)
    }.map(x => x._1 + " " + x._2)

    //training.take(10).foreach(println)

    training.coalesce(1).saveAsTextFile("E:/git/data/ch02/stage02")

    sc.stop()
  }
}
