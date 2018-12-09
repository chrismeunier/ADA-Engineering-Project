#############################################################################################
###################### Script 2, perform lda on transformed dataframes ######################
#############################################################################################

#import findspark
#findspark.init()

from pyspark.sql import *
from pyspark.ml.clustering import LDA, LDAModel

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

#Initialisation
alpha = 30
beta = 0.0
ntopics = 10
sample_fraction = 0.01
tfidf = spark.read.parquet('tfidf_all.parquet')
voc = sc.textFile('voc.txt').collect()


############## LDA stage ##############
#training
tfidf_sample = tfidf.sample(fraction=sample_fraction)
alpha_asymmetric = [alpha/(k+1) for k in range(ntopics)]
lda_model = LDA(k=ntopics, maxIter=100, docConcentration=alpha_asymmetric,topicConcentration=beta).setFeaturesCol('non_norm_features').fit(tfidf_sample)

#get topics and word list
topics = lda_model.describeTopics()

#get topic distribution for all texts
result_lda = lda_model.transform(tfidf).drop('non_norm_features')



############## Joining stage ##############
#get the source data
wlp_rdd = sc.textFile(WLP_FILE).zipWithIndex().filter(lambda r: r[1] > 2).keys()
topic_source_bytext = sources.join(result_lda,'textID','inner').drop('url','web','website','lemma_list','raw_features','features').sort('textID')

sources_rdd = sc.textFile('sample_data/now-samples-sources.txt')\
                .map(lambda r: r.split('\t'))

header = sources_rdd.take(3)
sources_rdd = sources_rdd.filter(lambda l: l != header[0])\
                .filter(lambda l: l != header[1])\
                .filter(lambda l: l != header[2])

                #create schema and change data type for date
sources_schema = sources_rdd.map(lambda r: Row(textID=int(r[0]),nwords=int(r[1]),date=r[2],country=r[3],website=r[4],url=r[5],title=r[6],))
sources = spark.createDataFrame(sources_schema)
sources = sources.withColumn('date',to_date(sources.date, 'yy-MM-dd'))
