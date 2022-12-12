package machinelearning

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.types._
import org.apache.spark.sql.Dataset


object Gbt{

  case class data_Energy(Date: String, lat: Double, lon: Double, Energy_restante: Double, Energy_consomme: Double) extends Serializable

  def main(args: Array[String]) {
    val modeldir="C:\\Users\\ibrah\\Downloads\\mapr-sparkml-sentiment-classification-master\\src\\modele\\gbt"
    val spark: SparkSession = SparkSession.builder().appName("flightdelay")
      .master("local[*]").getOrCreate()
    import spark.implicits._
    val schema = StructType(Array(
      StructField("Date", StringType, true),
      StructField("lat", DoubleType, true),
      StructField("lon", DoubleType, true),
      StructField("Energy_restante", DoubleType, true),
      StructField("Energy_consomme", DoubleType, true)

    ))
    val df: Dataset[data_Energy] = spark.read.option("inferSchema", "false")
      .schema(schema)
      .csv("C:\\Users\\ibrah\\Downloads\\mapr-sparkml-sentiment-classification-master\\src\\data\\data_Energy.csv")
      .as[data_Energy]
      df.show(5)
    //Creation new dataset
    val assembler = new VectorAssembler().
      setInputCols(df.drop("Date", "Energy_consomme").columns).
      setOutputCol("features")

    val df2 = assembler.transform(df)
    //Separation data en train et test
    val Array(train, test) = df2.randomSplit(Array(0.8, 0.2), seed = 100)
    //Creation du model machine learning
    val gbt = new GBTRegressor()
      .setLabelCol("Energy_consomme")
      .setFeaturesCol("features")
      .setMaxIter(90)
    //Application et ecriture du model
    val model = gbt.fit(train)
    model.write.overwrite().save(modeldir)

  }
}


