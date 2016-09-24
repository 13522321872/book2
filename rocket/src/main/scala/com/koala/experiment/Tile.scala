package com.koala.experiment

import breeze.linalg.DenseMatrix
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by seawalker on 2016/4/9.
 */
object Tile {
  def main(args: Array[String]) {
    val gpss = Seq((1.0, 2.0),(3.0,4.0),(5.0, 6.0))
    val elems = Array(1.0, 2.0, 3.0,4.0, 5.0, 6.0)
    val col1 = gpss.map(_._1).toArray
    val col2 = gpss.map(_._2).toArray

    val m = new DenseMatrix(3, 2, col1 ++ col2)

  }
}
