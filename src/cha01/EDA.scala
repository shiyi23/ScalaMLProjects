package cha01

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.AnalysisException

/**
  * This EDA02.scala to do the explore analysis to the dataset
  */

class EDA {

}



object EDA extends App {
  val train = "/home/huang/ComputerApplying" +
    "/ScalaMLProjects/dataset/train.csv"

  val spark = SparkSession
    .builder()
    .master("local") //设置程序要链接的Spark集群的master节点的URL，如果设置为local，则
    //  代表Spark程序在本地运行，特别适合机器配置非常差（例如内存只有1GB）的场景
    .appName("Allstate Insurance")
    .getOrCreate()
  import spark.implicits._ //to execute the implicits convertions

  val trainInput = spark.read
    .option("header","true")
    .option("inferSchema","true")
    .format("com.databricks.spark.csv")
    .load(train)
    .cache()

  val df = trainInput
  val newDF = df.withColumnRenamed("loss","label")

  newDF.createOrReplaceTempView("insurance")



  //  spark.sql("select avg(insurance.label) as AVG_LOSS from insurance").show()
  //
  //  spark.sql("select min(insurance.label) as MIN_LOSS from insurance").show()

}
