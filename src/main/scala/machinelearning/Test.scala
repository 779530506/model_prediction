package machinelearning

import org.apache.spark.sql.SparkSession

object Test extends  App {
  println("testing");
  var spark = SparkSession.builder().appName("test").master("local").getOrCreate();
  println("ok");

}
