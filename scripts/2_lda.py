#############################################################################################
###################### Script 2, perform lda on transformed dataframes ######################
#############################################################################################

import findspark
findspark.init()

from pyspark.sql import *
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.sql.types import DateType

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

#Initialisation
ALPHA = 3
BETA = 0.0
NTOPICS = 10
SAMPLE_FRACTION = 0.1
SOURCES_FILE = '/datasets/now_corpus/corpus/source/now_sources_pt*.txt'

tfidf = spark.read.parquet('tfidf_all.parquet')
voc = sc.textFile('voc.txt').collect()


############## LDA stage ##############
#training
tfidf_sample = tfidf.sample(fraction=SAMPLE_FRACTION)
alpha_asymmetric = [ALPHA/(k+1) for k in range(NTOPICS)]
lda_model = LDA(k=NTOPICS, maxIter=10, docConcentration=alpha_asymmetric,topicConcentration=BETA).setFeaturesCol('non_norm_features').fit(tfidf_sample)

#get topics and word list
topics = lda_model.describeTopics()

#get topic distribution for all texts
result_lda = lda_model.transform(tfidf).drop('features')



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

#save topic distribution over words
topics.write.mode('overwrite').parquet('topics.parquet')
