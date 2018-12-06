###############################################################
############ Script 0, importing data and cleaning it #########
###############################################################



import re
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.ml.feature import CountVectorizer , IDF

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext


wlp_file = '/datasets/now_corpus/corpus/wlp/*-*-*.txt'
bottom_percent = 0.8
top_percent = 0.95



############## Loading stage ##############
#first read the text file
wlp_rdd = sc.textFile(wlp_file).zipWithIndex().filter(lambda r: r[1] > 2).keys()

#the first 3 lines are useless headlines
header = wlp_rdd.take(3)
noheaders = wlp_rdd.filter(lambda r: r != header[0]).filter(lambda r: r != header[1]).filter(lambda r: r != header[2])

#we split the elements separated by tabs
lines = noheaders.map(lambda r: r.split('\t'))

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
[bottom,top] = lemma_freq.approxQuantile('count', [bottom_percent,top_percent], 0.01)
lemma_tokeep = lemma_freq.filter(lemma_freq['count']>bottom).filter(lemma_freq['count']<top)
c = lemma_tokeep.count()
print('Number of lemmas left: %'%(c))
print('Percentage of lemmas left: %.2f'%(c/lemma_freq.count()*100))

wlp_nostop.registerTempTable('wlp_nostop')
lemma_tokeep.registerTempTable('lemma_tokeep')

query = """
SELECT wlp_nostop.lemma, wlp_nostop.textID
FROM wlp_nostop
INNER JOIN lemma_tokeep ON wlp_nostop.lemma = lemma_tokeep.lemma
"""

wlp_kept = spark.sql(query)
wlp_bytext = wlp_kept.groupBy('textID').agg(collect_list('lemma')).sort('textID').withColumnRenamed('collect_list(lemma)','lemma_list')



############## TF-IDF stage ##############
cvmodel = CountVectorizer(inputCol="lemma_list", outputCol="raw_features").fit(wlp_bytext)
result_cv = cvmodel.transform(wlp_bytext)
idfModel = IDF(inputCol="raw_features", outputCol="features").fit(result_cv)
result_tfidf = idfModel.transform(result_cv).drop('lemma_list','raw_features')



############## Saving stage ##############
#saving dataframe
result_tfidf.write.mode('overwrite').parquet('wlp_tfidf_all.parquet')
#saving vocabulary from CountVectorizer
sc.parallelize(cvmodel.vocabulary).saveAsTextFile('voc.txt')
