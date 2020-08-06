package cha01

import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

/**
  * 此源代码.scala文件做了数据预处理的相关工作
  */

class Preprocessing {



}

object Preprocessing {

  var trainSample = 1.0

  var testSample = 1.0


  val train = "/home/huang/ComputerApplying" +
    "/ScalaMLProjects/dataset/train.csv"

  val test = "/home/huang/ComputerApplying" +
    "/ScalaMLProjects/dataset/test.csv"

  val spark = SparkSession
    .builder()
    .master("local")
    .appName(" Data Preprocessing")
    .getOrCreate()

  println("Reading data from" + train + " file")

  val trainInput = spark.read
    .option("header","true")
    .option("inferSchema","true")
    .format("com.databricks.spark.csv")
    .load(train)
    .cache()

  val testInput = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("com.databricks.spark.csv")
    .load(test)
    .cache()

  println("Preparing data for training model")

  var data = trainInput.withColumnRenamed("loss",
    "label").sample(false, trainSample)
  var DF = data.na.drop() //drop the null column

  if (data == DF) {
    println("no null values")
  } else {
    println("null values exist")
    data = DF
  }

  val seed = 12345L
  val splits = data.randomSplit(Array(0.75, 0.25),seed)
  val (trainingData, validationData) = (splits(0), splits(1))

  //缓存这两个数据集
  trainingData.cache()
  validationData.cache()

  val testData = testInput.sample(false, testSample).cache()

  def isCateg(c: String): Boolean = c.startsWith("cat")

  def categNewCol(c: String): String = if (isCateg(c) ) s"idx_${c}" else c

  def removeTooManyCategs(c: String): Boolean = !(c matches "cat(109$|110$112$113$|116$)")

  def onlyFeatureCols(c: String) : Boolean = !(c matches "id|label")

  val featureCols = trainingData.columns
    .filter(removeTooManyCategs)
    .filter(onlyFeatureCols)
    .map(categNewCol)

  var stringIndexerStages = trainingData.columns.filter(isCateg)
    .map(c => new StringIndexer()
      .setInputCol(c)
      .setOutputCol(categNewCol(c))
      .fit(trainInput.select(c).union(testInput.select(c))) )

  val assembler = new VectorAssembler()
    .setInputCols(featureCols)
    .setOutputCol("features")

}
