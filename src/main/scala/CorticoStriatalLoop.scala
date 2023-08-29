import CorticoStriatalLoop.Node
import weka.classifiers.AbstractClassifier
import weka.clusterers.SimpleKMeans
import weka.core.{EuclideanDistance, Instance, Instances, SelectedTag}

import scala.collection.mutable

class CorticoStriatalLoop extends AbstractClassifier{

  private var root: Node = null


  //Algorithm 1 from paper
  override def buildClassifier(data: Instances): Unit = {
    val q = mutable.Queue.empty[Node]
    root = Node(data, null, mutable.Seq())
    q.addOne(root)
    while(q.nonEmpty){
      val qn = q.dequeue()
      if(subdivide(qn.instances)){
        val maxK = 4
        val kMeans = partition(qn.instances, Math.min(qn.instances.numClasses(), maxK))
        qn.branches = kMeans.getClusterCentroids
        Range(0, kMeans.numberOfClusters()).foreach{u =>
          val childInstances = new Instances(qn.instances, qn.instances.numInstances())
          qn.instances.stream().filter(kMeans.clusterInstance(_) == u).forEach(v => childInstances.add(v))
          qn.children :+ Node(childInstances, null, mutable.Seq())
        }
      }
    }


    //If the data come from multiple classes, SUBDIVIDE returns true and otherwise, false.
    def subdivide(instances: Instances): Boolean = {
      instances.numClasses() > 1
    }


    //unsupervised clustering algorithm
    //unlike the paper, this just uses kMeans++ as we are not just dealing with real numbers
    def partition(instances: Instances, k: Int): SimpleKMeans ={
      val kMeans = new SimpleKMeans()
      kMeans.setInitializationMethod(new SelectedTag(SimpleKMeans.KMEANS_PLUS_PLUS, SimpleKMeans.TAGS_SELECTION))
      kMeans.setNumClusters(k)
      val classIndex = instances.classIndex()
      instances.setClassIndex(-1)
      kMeans.buildClusterer(instances)
      instances.setClassIndex(classIndex)
      kMeans
    }

  }


  //Algorithm 2 from paper
  override def classifyInstance(instance: Instance): Double = {
    var node: Node = root
    while(node.children.nonEmpty){
      var mostSim = 0.0
      var branchIndex: Int = -1
      val branches = node.branches.toArray(new Array[Instance](node.branches.numInstances()))
      val allInstances = node.branches
      node.branches.add(instance)
      val euclideanDistance = new EuclideanDistance(allInstances)
      Range(0, branches.length).foreach{u =>
        val sim = similarity(instance, branches(u), euclideanDistance)
        if(sim > mostSim){
          mostSim = sim
          branchIndex = u
        }
      }
      node = node.children(branchIndex)
    }


    def similarity(instance1: Instance, instance2: Instance, euclideanDistance: EuclideanDistance): Double ={
      1.0 / euclideanDistance.distance(instance1, instance2)
    }


    node.instances.firstInstance().classValue()
  }

}

object CorticoStriatalLoop{
  private case class Node(instances: Instances, var branches: Instances, children: mutable.Seq[Node])
}
