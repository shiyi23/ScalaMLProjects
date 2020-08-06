package cha01

import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{Row, SparkSession}
import cha01.Preprocessing.stringIndexerStages



class LRRegression {

}

object LRRegression extends App {
  val spark = SparkSession
    .builder()
    .master("local")
    .appName("MyFirstLRRegression")
    .getOrCreate()
  import spark.implicits._

  val numFolds = 10
  val MaxIter: Seq[Int] = Seq(1000)

  val RegParam: Seq[Double] = Seq(0.001)

  val Tol: Seq[Double] = Seq(1e-6)

  val ElasticNetParam: Seq[Double] = Seq(0.001)

  //创建一个线性回归估值器

  val model = new LinearRegression()
    .setFeaturesCol("features")
    .setLabelCol("label")

  //通过链接变换器和线性回归估计器来构建pipeline估计器

  println("正在建立ML pipeline...")
  val pipeline = new Pipeline()
    .setStages((Preprocessing.stringIndexerStages :+ Preprocessing.assembler)
      :+ model)

  /**
    * 通过指定最大迭代次数、回归参数、容错值和弹性网络参数来创建参数网格。
    * 而创建参数网格的目的是为了利用网格搜索来找到最优的超参数。
    *
    * 相关参考链接：https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/
    */
  val paramGrid = new ParamGridBuilder()
    .addGrid(model.maxIter,MaxIter)
    .addGrid(model.regParam, RegParam)
    .addGrid(model.tol, Tol)
    .addGrid(model.elasticNetParam, ElasticNetParam)
    .build()


  println("进行10折交叉认证，并用网格搜索来调整模型")

  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(new RegressionEvaluator())
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds)

  println("开始训练线性回归模型")

  val cvModel = cv.fit(Preprocessing.trainingData)

  println("在训练集和验证集上评估模型，并算出5个度量回归误差的指标")

  val trainPredictionsAndLabels =
    cvModel.transform(Preprocessing.trainingData)
    .select("label", "prediction")
    .map{case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

  val validPredictionAndLabels =
    .cvModel.transform()


}
