package stream

// http://maprdocs.mapr.com/home/Spark/Spark_IntegrateMapRStreams.html

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.streaming._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans

/**
 * Consumes messages from a topic in MapR Streams using the Kafka interface,
 * enriches the message with  the k-means model cluster id and publishs the result in json format
 * to another topic
 * Usage: SparkKafkaConsumerProducer  <model> <topicssubscribe> <topicspublish>
 *
 *   <model>  is the path to the saved model
 *   <topic> is a  topic to consume from
 *   <tableName> is a table to write to
 * Example:
 *    $  spark-submit --class com.sparkkafka.uber.SparkKafkaConsumerProducer --master local[2] \
 * mapr-sparkml-streaming-uber-1.0.jar /user/user01/data/savemodel  /user/user01/stream:ubers /user/user01/stream:uberp
 *
 *    for more information
 *    http://maprdocs.mapr.com/home/Spark/Spark_IntegrateMapRStreams_Consume.html
 */

object StructuredStreamingConsumer extends Serializable {

  case class Uber(dt: String, lat: Double, lon: Double, base: String) extends Serializable
  def main(args: Array[String]): Unit = {

    val modeldirectory = "C:\\Users\\ibrah\\Downloads\\mapr-sparkml-sentiment-classification-master\\src\\modele"
    val spark: SparkSession = SparkSession.builder().appName("stream").master("local[*]").getOrCreate()
    import spark.implicits._
    val schema = StructType(Array(
      StructField("dt", TimestampType, true),
      StructField("lat", DoubleType, true),
      StructField("lon", DoubleType, true),
      StructField("base", StringType, true)
    ))
    // Spark 2.1
    val df = spark.read.option("inferSchema", "false").schema(schema).csv("C:\\Users\\ibrah\\Downloads\\mapr-sparkml-sentiment-classification-master\\src\\data\\uber-split2.csv").as[Uber]
df.show
    df.cache
    val featureCols = Array("lat", "lon")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(df)
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), 5043)
    // increase the iterations if running on a cluster (this runs on a 1 node sandbox)
    val kmeans = new KMeans().setK(4).setFeaturesCol("features").setMaxIter(5)
    val model = kmeans.fit(trainingData)
    //model.clusterCenters.foreach(println)
    val categories = model.transform(testData)
    //categories.show
    // to save the model
    model.write.overwrite().save(modeldirectory)

    val ds =spark.readStream.format("kafka")
      .option("kafka.bootstrap.servers", "51.77.212.74:9092")
      .option("subscribe", "intel1")
      .option("group.id", "testgroup")
      .option("startingOffsets", "earliest")
      .load()

    val model1 = org.apache.spark.ml.clustering.KMeansModel.load(modeldirectory)
    val ds2 = ds.select($"value" cast "string" as "json").select(from_json($"json", schema) as "data").select("data.*")
    val ds3=model1.transform(assembler.transform(ds2))
      //.withColumnRenamed("prediction", "cluster")
      //.drop("features")
      .select(to_json(struct("*")).as("value"))


    ds3.writeStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "51.77.212.74:9092")
      .option("topic", "intel2")
      .option("checkpointLocation", "/toto/test123222")
      .start()
    spark.streams.awaitAnyTermination()

  }
}