package com.koala.ch02

import org.apache.log4j.{Logger,Level}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time.format.DateTimeFormat
import org.joda.time.{DateTime, LocalDate}
;
/**
 * Created by haixiang on 2016/4/9.
 * exploring the gowalla data set.
 * download url: https://snap.stanford.edu/data/loc-gowalla.html
 * format: [user]	[check-in time]		[latitude]	[longitude]	[location id]
 * size: 6442892
 * user: 107092
 */
case class CheckIn(user: String, time: DateTime, latitude: Double, longitude: Double, location: String)

object GowallaDatasetExploration {

  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val Array(input, mode) = args
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName).setMaster(mode)
    val sc = new SparkContext(conf)

    val gowalla = sc.textFile(input).map(_.split("\t")).mapPartitions{
      case iter =>
        val format = DateTimeFormat.forPattern("yyyy-MM-dd\'T\'HH:mm:ss\'Z\'")
        iter.map{
        case terms => CheckIn(terms(0), DateTime.parse(terms(1),format), terms(2).toDouble, terms(3).toDouble,terms(4))
      }
    }

    //user, check ins, check in days, locations
    val observations = gowalla.map{
      case check => (check.user, (1L, Set(check.time.toLocalDate), Set(check.location)))
    }.reduceByKey{
      case (left, right) => (left._1 + right._1, left._2.union(right._2),  left._3.union(right._3))
    }.map{
      case (user, (checkins, days:Set[LocalDate], locations:Set[String])) =>
        Vectors.dense(checkins.toDouble, days.size.toDouble, locations.size.toDouble)
    }


    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    println(summary.mean)
    println(summary.variance)
    println(summary.numNonzeros)

    sc.stop()

  }
}
