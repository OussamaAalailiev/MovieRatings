package org.enset.iibdcc

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.LoggerFactory

import scala.util.Random

object MovieLensRatingALS {
  final val LOGGER = LoggerFactory.getLogger(MovieLensRatingALS.getClass.getName)

  def main(args: Array[String]): Unit = {
    require(args.length > 1, "Error, Arguments not found!")
    /**Configuring the spark environment */
    val jarFile = "target/scala-2.11/movielensals_2.11-0.1.jar"
    val conf = new SparkConf()
      .setAppName(MovieLensRatingALS.getClass.getName).setMaster("local[*]")
      .set("spark.executor.memory", "8g")
      .setJars(Seq(jarFile))
    val sc = new SparkContext(conf)
    //Modifying the path in here to an HDFS Path down below:
  val ratings = sc.textFile(args(0)).map { line =>
 // val ratings = sc.textFile("hdfs://hadoop-master:9000/user/root/data/movies-input/ratings.dat").map { line =>
      val fields = line.split("::")
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }
    //Modifying the path in here to an HDFS Path down below:
    val movies = sc.textFile("src/main/resources/movies.dat").map { line =>
//  val movies = sc.textFile("hdfs://hadoop-master:9000/user/root/data/movies-input/movies.dat").map { line =>
      val movieFields = line.split("::")
      (movieFields(0).toInt, movieFields(1))
    }.collect.toMap
    /**My code here, I should clean data i think!*/
    /**Counting the number of ratings, users and movies*/

    val numRatings = ratings.count()
    val numUsers = ratings.map(_._2.user).distinct.count
    val numMovies = ratings.map(_._2.product).distinct.count
    println("We got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies. ")
    /**To make recommendation for you, we will extract top 50 movies rated then*/
    /**we will ask you to do some rating, later we will return your ratings..*/
    val mostRatedMoviesIds = ratings.map(_._2.product).countByValue().toSeq
      .sortBy(-_._2).take(50).map(_._1)
    val random = new Random(0)
    val selectedMovies = mostRatedMoviesIds.filter(x => random.nextDouble() < 0.2)
      .map(x => (x, movies(x)))

    /** Elicitate ratings from command-line. */
    def elicitateRatings(movies: Seq[(Int, String)]) = {
      val prompt = "Rate the following movies (1-5 points)"
      println(prompt)
      val ratings = movies.flatMap { x =>
        var rating: Option[Rating] = None
        var valid = false
        while (!valid) {
          print(x._2 + ": ")
          try {
            val r = Console.readInt
            if (r < 0 || r > 5) {
              println(prompt)
            } else {
              valid = true
              if (r > 0) {
                rating = Some(Rating(0, x._1, r))
              }
            }
          } catch {
            case e: Exception => println(prompt)
          }
        }
        rating match {
          case Some(r) => Iterator(r)
          case None => Iterator.empty
        }
      }
      if (ratings.isEmpty) {
        error("No rating provided!")
      } else {
        ratings
      }

    }
    val myRatings = elicitateRatings(selectedMovies)
    val myRatingsRDD = sc.parallelize(myRatings)
    /**Splitting DataSets into 3 subset of Data : "Training Set 60%", "Validation Set 20%" and "Test Set 20%" */
    val numPartitions = 20
      val training = ratings.filter(x=> x._1 < 6)
        .values.union(myRatingsRDD).repartition(numPartitions).persist
      val validation = ratings.filter(x=> x._1>=6 && x._1 <8)
        .values.repartition(numPartitions).persist
      val test = ratings.filter(x=> x._1>=8)
        .values.persist
     //Counting the number of training, validation and test Sets down below :
      val numTraining = training.count()
      val numValidation = validation.count()
      val numTest = test.count()
    println("Training: " + numTraining + " , Validation:" + numValidation + " , Test: " +numTest)
    /** Compute RMSE (Root Mean Squared Error). */
    def computeRMSE(model:MatrixFactorizationModel, data:RDD[Rating], n:Long): Double = {
      val predictions :RDD[Rating] = model.predict(data.map(x=> (x.user,x.product)))
      val predictionsAndRatings = predictions.map(x=>((x.user,x.product),x.rating))
        .join(data.map(x=>((x.user,x.product),x.rating))).values
      math.sqrt(predictionsAndRatings.map(x=>(x._1 - x._2)*(x._1 - x._2)).reduce(_+_)/n)
    }
    /**Training the models by ALS Algorithm + Defining the parameters of ALS */
    val ranks = List(8,12)
    val lambdas = List(0.1, 10.0)
    val numIters = List(10, 20)
    var bestModel:Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for(rank <-ranks; lambda<-lambdas; numIter<-numIters){
      val model = ALS.train(training, rank, numIter, lambda)
      val validationRmse = computeRMSE(model, validation, numValidation)
      println("RMSE (Validation) = " + validationRmse + " for the model trained with rank = "
        + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if(validationRmse<bestValidationRmse){
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }
    val testRmse = computeRMSE(bestModel.get, test, numTest)
    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
           + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse +".")

    /**let’s take a look at what movies our model recommends for you.*/
    val myRatedMovieIds = myRatings.map(_.product).toSet
    val candidates = sc.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq)
    val recommendations  = bestModel.get
      .predict(candidates.map((0, _)))
      .collect
      .sortBy(-_.rating)
      .take(50)


    var i =1
    println("Movies recommended for you : ")
    recommendations.foreach{
      r => println("%2d".format(i) + ": " + movies(r.product))
        i+=1
    }
    /**Does ALS output a non-trivial model? let's see: */
    val meanRating = training.union(validation).map(_.rating).mean
    val baselineRMSE = math.sqrt(test.map(x=>(meanRating - x.rating)
                                * (meanRating - x.rating)).reduce(_+_) / numTest)
    val improvement = (baselineRMSE - testRmse) / baselineRMSE*100
    println("The best model improves the baseline by = " + "%1.2f".format(improvement) + "%.")

    /**I need to save the results before sending it to HDFS then execute it in there*/
  }
}
