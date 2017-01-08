package com.koala.ch03

import com.koala.util.AppConst
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by haixiang on 2016/9/24.
  */

object AppTrainingData {

  def main(args: Array[String]): Unit = {
    /**
      * input: participles data
      * output: libsvm data.
      * mode: yarn-client, yarn-cluster or local[*].
      */
    val Array(input, output, mode) = args
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val sc = new SparkContext(new SparkConf().setMaster(mode).setAppName(this.getClass.getName))

    val rdd = sc.textFile(input)
      .map(_.split("~", -1))
      .map{
        case terms =>
          //package name, app name, class, key words, introduction.
          (terms(0), terms(1), terms(2), terms(3), terms(4))
      }.map {
      case (panme, aname, c, kw, intro) =>
        val flt = intro.split(" ").map(_.split("/")).filter(x => x(0).length > 1 && filterProp(x(1))).map(x => x(0))
        (panme, aname, c, flt )
    }.map(x => (x._3, x._4))


    val minDF = rdd.flatMap(_._2.distinct).distinct()
    val indexes = minDF.collect().zipWithIndex.toMap
    val training = rdd.repartition(4).map{
      case (label, terms) =>
        val svm = terms.map(v => (v, 1)).groupBy(_._1).map {
          case (v, vs) => (v, vs.length)
        }.map{
          case (v, cnt) => (indexes.get(v).getOrElse(-1) + 1, cnt)
        }.filter(_._1 > 0)
          .toSeq
          .sortBy(_._1)
          .map(x => "" + x._1 + ":" + x._2)
          .mkString(" ")
        (AppConst.APP_CLASSES.indexOf(label), svm)
    }.filter(!_._2.isEmpty)
      .map(x => "" + x._1 + " " + x._2)

    training.coalesce(1).saveAsTextFile(output)

    sc.stop()
  }

  def filterProp(p:String):Boolean = {
    p.equals("v") || p.contains("n")
  }

}
