package com.koala.experiment

import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by seawalker on 2016/4/9.
 */
object ReBroadcast {
  val gps = Seq((1.0, 2.0),(3.0,4.0),(5.0, 6.0))

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName(this.getClass.getSimpleName).setMaster("local")
    val sc = new SparkContext(conf)

  }

  def batch(sc:SparkContext)={
    val rdd = sc.parallelize(gps)
    var tagb = sc.broadcast("01")
    rdd.foreach(x => print(tagb.value + "\t "))
    tagb = sc.broadcast("02")
    rdd.foreach(x => print(tagb.value + "\t "))
    sc.stop()
  }

  def streaming(sc:SparkContext) ={
    //TODO
  }
}
