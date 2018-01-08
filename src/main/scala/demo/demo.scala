import java.io.File
import opennlp.tools.sentdetect.{SentenceDetectorME,SentenceModel}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object SentenceDetector_Demo {
def main(args:Array[String]): Unit ={
val conf = new SparkConf()
.setAppName("SentenceDetector_Application")
.setMaster("spark:master:7077")
.set("spark.serializer",
"org.apache.spark.serializer.KryoSerializer")
val sc = new SparkContext(conf)
val textInput = sc.makeRDD(Array("C-122 co-stimulates T cells via Aiolos and Ikaros degradation.",
"Aiolos and Ikaros have been described asrepressors of IL-2 expression and secretion, a marker of activation in primary T-cells 13,21,27. ",
"Given thedegradation observed of Aiolos and Ikaros in DLBCL cells, we investigated the",
"effects of CC-122 treatment on IL-2 expression in primary T-cells."),1)
val sentenceDetectorModelFile = new File("/home/opennlp_models/en-sent.bin")
val model = new SentenceModel(sentenceDetectorModelFile)
val sdetector = new SentenceDetectorME(model)
val broadCastedsdector = sc.broadcast(sdetector)
val broadCastedsdector = sc.broadcast(sdetector)
val results = textInput.map{record =>
(broadCastedsdector.value.sentDetect(record),
broadCastedsdector.value.getSentenceProbabilities)
}
val detectedSentences = results.keys.flatMap(x => x)
val probabilities = results.values.flatMap(x => x)
println("Detected Sentences: ")
detectedSentences.collect().foreach(println)
println("Probabilities : ")
probabilities.collect().foreach(println) }}