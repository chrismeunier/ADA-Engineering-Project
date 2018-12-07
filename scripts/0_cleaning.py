###############################################################
############ Script 0, importing data and cleaning it #########
###############################################################



import re
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.ml.feature import CountVectorizer , IDF, Normalizer



spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext


WLP_FILE = '/datasets/now_corpus/corpus/wlp/*-*-*.txt'
BOTTOM_PERCENT= 0.8
TOP_PERCENT = 0.95



############## Loading stage ##############
#read the text file and remove the first three rows (zip trick)
wlp_rdd = sc.textFile(WLP_FILE).zipWithIndex().filter(lambda r: r[1] > 2).keys()

#we split the elements separated by tabs
lines = wlp_rdd.map(lambda r: r.split('\t'))

#identify the columns
wlp_schema = lines.map(lambda r: Row(textID=int(r[0]),idseq=int(r[1]),word=r[2],lemma=r[3],pos=r[4]))
wlp = spark.createDataFrame(wlp_schema)



############## Cleaning stage ##############
#pos
pos_remove = ['.',',',"\'",'\"','null']
wlp_nopos = wlp.filter(~wlp['pos'].isin(pos_remove)).filter(~wlp['pos'].startswith('m')).filter(~wlp['pos'].startswith('f')).drop('idseq','pos','word')

#stopwords
stopwords = sc.textFile('our_stopwords.txt').collect()
wlp_nostop = wlp_nopos.filter(~wlp['lemma'].isin(stopwords))
lemma_freq = wlp_nostop.groupBy('lemma').count().sort('count', ascending=False)

#removal
[bottom,top] = lemma_freq.approxQuantile('count', [BOTTOM_PERCENT,TOP_PERCENT], 0.01)
lemma_tokeep = lemma_freq.filter(lemma_freq['count']>bottom).filter(lemma_freq['count']<top)
print('Number of lemmas left: %d'%lemma_tokeep.count())

wlp_nostop.registerTempTable('wlp_nostop')
lemma_tokeep.registerTempTable('lemma_tokeep')

query = """
SELECT wlp_nostop.lemma, wlp_nostop.textID
FROM wlp_nostop
INNER JOIN lemma_tokeep ON wlp_nostop.lemma = lemma_tokeep.lemma
"""

wlp_kept = spark.sql(query)
wlp_bytext = wlp_kept.groupBy('textID').agg(collect_list('lemma')).sort('textID').withColumnRenamed('collect_list(lemma)','lemma_list')
print('Number of documents: %d'%wlp_bytext.count())



############## TF-IDF stage ##############
#tf
cvmodel = CountVectorizer(inputCol="lemma_list", outputCol="raw_features").fit(wlp_bytext)
result_cv = cvmodel.transform(wlp_bytext)

#idf
idfModel = IDF(inputCol="raw_features", outputCol="non_norm_features").fit(result_cv)
result_tfidf = idfModel.transform(result_cv).drop('lemma_list','raw_features')

#normalised (by default euclidean norm)
norm = Normalizer(inputCol="non_norm_features", outputCol="features")
tfidf_norm = norm.transform(result_tfidf).drop('non_norm_features')



############## Saving stage ##############
#saving dataframe
tfidf_norm.write.mode('overwrite').parquet('tfidf_all.parquet')
#saving vocabulary from CountVectorizer
sc.parallelize(cvmodel.vocabulary).saveAsTextFile('voc.txt')
