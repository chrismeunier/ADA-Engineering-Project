##########################################################################################################
###################### Script 3, selection fo the informations we want to download ######################
#########################################################################################################

#import findspark
#findspark.init()

from pyspark.sql import *
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.sql.types import DateType

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

#Initialisation
SOURCES_FILE = '/datasets/now_corpus/corpus/source/now_sources_pt*.txt'

tfidf = spark.read.parquet('tfidf_all.parquet')
result_lda = spark.read.parquet('result_lda.parquet')
voc = sc.textFile('voc.txt').collect()


############## Joining stage ##############
#read the source data and split by tabs
sources_rdd = sc.textFile(SOURCES_FILE).zipWithIndex().filter(lambda r: r[1] > 2).keys().map(lambda r: r.split('\t'))

#create schema and change data type for date
sources_schema = sources_rdd.map(lambda r: Row(textID=int(r[0]),nwords=int(r[1]),date=r[2],country=r[3],website=r[4],url=r[5],title=r[6],))
sources = spark.createDataFrame(sources_schema)

#join topic distribution with sources
topic_source_bytext = sources.join(result_lda,'textID','inner').drop('url','web','website','title')



############## Saving stage ##############
#save topic distribution
topic_source_bytext.write.mode('overwrite').parquet('topic_source_bytext.parquet')
