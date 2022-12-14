package machinelearning
import org.apache.spark.sql.types._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans

object Kmeans {

    case class Uber(dt: String, lat: Double, lon: Double, base: String) extends Serializable

    def main(args: Array[String]) {

      val spark: SparkSession = SparkSession.builder().appName("flightdelay")
                               .master("local[*]").getOrCreate()
      import spark.implicits._
      val schema = StructType(Array(
        StructField("dt", TimestampType, true),
        StructField("lat", DoubleType, true),
        StructField("lon", DoubleType, true),
        StructField("base", StringType, true)
      ))
      // Spark 2.1
      val df: Dataset[Uber] = spark.read.option("inferSchema", "false")
                                         .schema(schema)
                                          .csv("C:\\Users\\ibrah\\Downloads\\mapr-sparkml-sentiment-classification-master\\src\\data\\uber.csv").as[Uber]

      df.cache
      val featureCols = Array("lat", "lon")
      val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
      val df2 = assembler.transform(df)
      val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), 5043)
      // increase the iterations if running on a cluster (this runs on a 1 node sandbox)
      val kmeans = new KMeans().setK(20).setFeaturesCol("features").setMaxIter(5)
      val model = kmeans.fit(trainingData)
      model.write.overwrite().save("/user/user01/data/savemodel")
      model.clusterCenters.foreach(println)
      val categories = model.transform(testData)
      //categories.show
      model.write.overwrite().save("/tmp/modele")



  }
}

