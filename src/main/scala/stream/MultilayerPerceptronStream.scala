package stream
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{ Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
object MultilayerPerceptronStream {

  def main(args: Array[String]) {
    val modeldir="/home/abdoulayesarr/Documents/MasterBigdata/memoire/code/mapr-sparkml-sentiment-classification-master/src/modele/MultilayerPerceptron"
    val spark: SparkSession = SparkSession.builder().appName("flightdelay")
      .master("local[*]").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    import spark.implicits._

    val schema = StructType(Array(
      StructField("thickness", DoubleType, true),
      StructField("size", DoubleType, true),
      StructField("shape", DoubleType, true),
      StructField("madh", DoubleType, true),
      StructField("epsize", DoubleType, true),
      StructField("bnuc", DoubleType, true),
      StructField("bchrom", DoubleType, true),
      StructField("nNuc", DoubleType, true),
      StructField("mit", DoubleType, true),
      StructField("user", StringType, true),
      StructField("nom_patient", StringType, true),
      StructField("prenom_patient", StringType, true),
      StructField("telephone_patient", StringType, true),
      StructField("lat", StringType, true),
      StructField("lon", StringType, true),
      StructField("ville", StringType, true)
    ))
   // {"medecin_user":"asarr050622","lat":14.7645042 ,"lon":-17.3660286,"nom_patient":"DIATTARA", "prenom_patient":"Ibrahima", "telephone_patient":"00221771445645", "ville":"dakar","thickness":5.0,"size":1.0,"shape":1.0,"madh":1.0,"epsize":2.0,"bnuc":3.0,"bchrom":1.0,"nNuc":1.0,"mit":2.0}
    val df = spark.createDataFrame(spark.sparkContext
      .emptyRDD[Row], schema)
    val assembler = new VectorAssembler().
      //setInputCols(df.drop("id","prenom","nom", "dateNaiss", "ville", "lon", "lat").columns)
      setInputCols(df.select("thickness", "size", "shape", "madh", "epsize", "bnuc", "bchrom", "nNuc", "mit").columns)
      .setOutputCol("features")
    //{"lat":14.7645042 ,"lon":-17.3660286,"nom":"DIATTARA", "prenom":"Ibrahima", "dateNaiss":"17-08-922", "ville":"dakar","thickness":5.0,"size":1.0,"shape":1.0,"madh":1.0,"epsize":2.0,"bnuc":3.0,"bchrom":1.0,"nNuc":1.0,"mit":2.0}
    val dfstreaming =spark.readStream.format("kafka")
      .option("kafka.bootstrap.servers", "51.77.212.74:9085")
      .option("subscribe", "Cancer_MultilayerPerceptron_Source")
      .option("group.id", "spark_Cancer_MultilayerPerceptron")
      .option("startingOffsets", "latest")
      .load()
    val modelload=org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel.load(modeldir)
    val df2s = dfstreaming.select($"value" cast "string" as "json").select(from_json($"json", schema) as "data").select("data.*")
    val df3s=modelload.transform(assembler.transform(df2s))
      .withColumn("agent_timestamp", date_format(current_timestamp(),"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))
      .withColumn("location", concat($"lat", lit(','), $"lon"))
      .drop("features")
      .select(to_json(struct("*")).as("value"))
    df3s.writeStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "51.77.212.74:9085")
      .option("topic", "Cancer_MultilayerPerceptron_Result")
      .option("checkpointLocation", "/tmp/Cancer_MultilayerPerceptron_Result")
      .start()
    spark.streams.awaitAnyTermination()
  }
}
