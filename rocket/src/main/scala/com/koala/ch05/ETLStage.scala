package com.koala.ch05

import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


/**
  * Created by seawalker on 2016/11/23.
  */

case class Sale(userId:Int,           //0
                Style:String,         //1
                Price:String,         //2
                Rating:Double,        //3
                Size:String,          //4
                Season:String,        //5
                NeckLine:String,      //6
                SleeveLength:String,  //7
                waiseline:String,     //8
                Material:String,      //9
                FabricType:String,    //10
                Decoration:String,    //11
                PatternType:String,	  //12
                Recommendation:Int,   //13
                MeanSale:Int              //14
               )

object ETLStage {

  val KEY_INDEXS = Map(
    "Style" -> 0,
    "Price" -> 1,
    "Rating" -> 2,
    "Size" -> 3,
    "Season" -> 4
  )

  val ONE_HOT_ENCODER = Seq(
    //style
    Map("brief" -> 1,"casual" -> 2,"vintage" -> 3,"novelty" -> 4,"ol" -> 5,"cute" -> 6,"work" -> 7,"bohemian" -> 8,"flare" -> 9,"sexy" -> 10,"party" -> 11,"fashion" -> 12),
    //price
    Map("high" -> 13,"very-high" -> 14,"medium" -> 15,"low" -> 16,"average" -> 17),
    //rating
    Map("default" -> 18), // 18
    //size
    Map("free" -> 19,"s" -> 20,"m" -> 21,"xl" -> 22,"l" -> 23),
    //season
    Map("autumn" -> 24,"spring" -> 25,"winter" -> 26, "summer" -> 27, "unknown" -> 28)
    //omitted
  )

  def etl(spark:SparkSession, salesPath:String):RDD[String] = {
    //import spark.implicits._
    val sc = spark.sparkContext
    val sales = sc.textFile(salesPath)
      .map {
        x =>
          val Array(a, b) = x.split("~")
          val c = b.split(" ", -1).map(_.toLowerCase)
          Sale(a.toInt,
            c(0), c(1), c(2).toDouble, c(3), c(4), c(5), c(6), c(7), c(8), c(9), c(10), c(11), c(12).toInt, c(13).toInt)
      }

    // handling missing or wrong values

    sales.map {
      sale =>
        val style = ("Style", sale.Style)
        val price = ("Price", sale.Price match {
          case "null" => "medium"
          case whoa => sale.Price
        })

        val rating = ("Rating", sale.Rating.toString)
        val size = ("Size", sale.Size match {
          case "small" => "s"
          case whoa => sale.Size
        })
        val season = ("Season", sale.Season match {
          case "automn" => "autumn"
          case "null" => "unknown"
          case "" => "unknown"
          case whoa => sale.Season
        })
        val label = sale.MeanSale
        val vector = Seq(style, price, rating, size, season).map {
          case (name, item) =>
            val m = ONE_HOT_ENCODER(KEY_INDEXS(name))
            m.size match {
              case 1 => (m("default"), item)
              case whoa => (m(item), 1)
            }
        }.sortBy(_._1)
          .map(x => x._1 + ":" + x._2).mkString(" ")
        Math.log(label) + " " + vector
    }
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val spark = SparkSession.builder
      .config("spark.sql.warehouse.dir", "")
      .master("local[1]")
      .appName(this.getClass.getName)
      .getOrCreate()
    //TODO omitted
    val basePath = "....."
    etl(spark,  basePath + "dresses_sales.txt")
     .coalesce(1).saveAsTextFile(basePath + "dresses_libsvm")

    spark.stop()
  }

//  val base = 23
//  val a = rdd.map(_.Season.toLowerCase).distinct().collect().zipWithIndex.map{
//    case (item, index) =>
//      val sIndex = base + index
//      s""""$item" -> $sIndex"""
//  }.mkString(",")
//  println (a)

}
