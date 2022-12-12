package machinelearning
  import org.apache.spark.ml.feature.VectorAssembler
  import org.apache.spark.sql.{SparkSession}
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types._
  import org.apache.spark.ml.feature.StringIndexer
  import org.apache.spark.sql.Column

  import org.apache.spark.ml.classification.MultilayerPerceptronClassifier

object MultilayerPerceptron {

  def func(column: Column) = column.cast(DoubleType)

  def main(args: Array[String]) {
    val datadir = "/home/abdoulayesarr/Documents/MasterBigdata/memoire/code/mapr-sparkml-sentiment-classification-master/src/data/cancer.csv"
    val modeldir = "/home/abdoulayesarr/Documents/MasterBigdata/memoire/code/mapr-sparkml-sentiment-classification-master/src/modele/MultilayerPerceptron"
    val spark: SparkSession = SparkSession.builder().appName("flightdelay")
      .master("local[*]").getOrCreate()
    import spark.implicits._
    val df = spark.read.option("header", true)
      .option("InferSchema", true).csv(datadir)
      .filter($"bnuc" =!= "?")
      .withColumn("bnuc", $"bnuc".cast(DoubleType))
      .withColumn("class", when($"class" === 2, lit(0)).otherwise(lit(1)))
      .drop("id")
    val obsDF = df.select(df.columns.map(c => func(col(c))): _*)

    //define the feature columns to put in the feature vector
    val featureCols = Array("thickness", "size", "shape", "madh", "epsize", "bnuc", "bchrom", "nNuc", "mit")
    //set the input and output column names
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    //return a dataframe with all of the  feature columns in  a vector column
    val df2 = assembler.transform(obsDF)
    //  Create a label column with the StringIndexer
    val labelIndexer = new StringIndexer().setInputCol("class").setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2)
    //  split the dataframe into training and test data
    val splitSeed = 5043
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)

    // specify layers for the neural network:
    // input layer of size 9 (features), two intermediate of size 5 and 4
    // and output of size 2 (classes)
    val layers = Array[Int](9, 5, 12, 2)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
    //.setBlockSize(128)
    //.setSeed(1234L)
    //.setMaxIter(100)
    val model = trainer.fit(trainingData)
    model.write.overwrite().save(modeldir)

  }
}
