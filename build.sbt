ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.11"

lazy val root = (project in file("."))
  .settings(
    name := "MachineLearningAlgorithmZoo"
  )

libraryDependencies += "nz.ac.waikato.cms.weka" % "weka-stable" % "3.8.6"
