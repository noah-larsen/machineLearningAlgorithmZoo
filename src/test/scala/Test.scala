import weka.classifiers.pmml.consumer.NeuralNetwork
import weka.classifiers.rules.DecisionTable
import weka.core.Instances

import scala.io.Source
import scala.jdk.CollectionConverters._

object Test extends App {

  val prefix = "/UCI/"
  val suffix = ".arff"
  val datasetNames = Seq("anneal.ORIG", "anneal", "audiology", "autos", "balance-scale", "breast-cancer", "breast-w",
    "colic.ORIG", "colic", "credit-a", "credit-g", "diabetes", "glass", "heart-c", "heart-h", "heart-statlog",
    "hepatitis", "hypothyroid", "ionosphere", "iris", "kr-vs-kp", "labor", "letter", "lymph", "mushroom",
    "primary-tumor", "segment", "sick", "sonar", "soybean", "splice", "vehicle", "vote", "vowel", "waveform-5000",
    "zoo"
  )
  val classifiers = Seq(new CorticoStriatalLoop, new DecisionTable)
  datasetNames.foreach{u =>
    val instances = new Instances(Source.fromURL(getClass.getResource(prefix + u + suffix)).reader())
    instances.setClassIndex(instances.numAttributes() - 1)
    classifiers.foreach{v =>
      val nFolds = 10
      val averageAccuracy = Range(0, nFolds).map{w =>
        v.buildClassifier(instances.trainCV(nFolds, w))
        val testInstances = instances.testCV(nFolds, w)
        testInstances.iterator().asScala.map { x =>
          v.classifyInstance(x) match {
            case y if y == x.classValue() => 1
            case _ => 0
          }
        }.sum / testInstances.numInstances().toDouble
      }.sum / nFolds
      System.out.println(u + " " + v.getClass.getName + " " + averageAccuracy)
    }


  }

}
