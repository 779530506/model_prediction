package stream

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.{OutputMode, Trigger}
import org.apache.spark.sql.types._

import java.util.TimeZone
import java.util.concurrent.TimeUnit
object Gbt{

  case class data_Energy(Date: String, lat: Double, lon: Double, Energy_restante: Double, Energy_consomme: Double) extends Serializable

  def main(args: Array[String]) {
    val modeldir="C:\\Users\\ibrah\\Downloads\\mapr-sparkml-sentiment-classification-master\\src\\modele\\gbt"
 val spark: SparkSession = SparkSession.builder().appName("flightdelay")
      .master("local[*]").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")


    import spark.implicits._
    val schema = StructType(Array(
      StructField("Date", StringType, true),
      StructField("lat", DoubleType, true),
      StructField("lon", DoubleType, true),
      StructField("Energy_restante", DoubleType, true),
      StructField("Energy_consomme", DoubleType, true)

    ))
    val df = spark.createDataFrame(spark.sparkContext
      .emptyRDD[Row], schema)
    val assembler = new VectorAssembler().
      setInputCols(df.drop("Date", "Energy_consomme").columns).
      setOutputCol("features")

    val dfstreaming =spark.readStream.format("kafka")
      .option("kafka.bootstrap.servers", "51.77.212.74:9092")
      .option("subscribe", "senelec_electricty_gbt_source")
      .option("group.id", "spark_senelec_electricty_gbt_source")
      .option("startingOffsets", "latest")
      .load()

    val modelload=org.apache.spark.ml.regression.GBTRegressionModel.load(modeldir)
    val df2s = dfstreaming.select($"value" cast "string" as "json").select(from_json($"json", schema) as "data").select("data.*")
    val df3s=modelload.transform(assembler.transform(df2s))
      .withColumn("agent_timestamp", date_format(current_timestamp(),"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))
      .withColumn("location", concat($"lat", lit(','), $"lon"))
      .drop("features", "lon", "lat", "Energy_consomme")
      .select(to_json(struct("*")).as("value"))
    df3s.writeStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "51.77.212.74:9092")
      .option("topic", "senelec_electricty_gbt_result_ml")
      .option("checkpointLocation", "/tata/senelec_electricty_gbt_result_mlZ")
      .start()
      spark.streams.awaitAnyTermination()
  }
}


