######################################################################################
################## Script 0, importing data, cleaning it and saving it ###############
######################################################################################

#import findspark
#findspark.init()

import re
from pyspark.sql import *
from pyspark.sql.functions import *

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

#Initialisation
WLP_FILE = '/datasets/now_corpus/corpus/wlp/*-*-*.txt'
#WLP_FILE = '../*-*-*.txt'
NPARTITION = 40
TOP_PERCENT = 0.95



############## Loading stage ##############
#read the text file and remove the first three rows (zip trick)
wlp_rdd = sc.textFile(WLP_FILE).zipWithIndex().filter(lambda r: r[1] > 2).keys()

#we split the elements separated by tabs
lines = wlp_rdd.map(lambda r: r.split('\t'))

#identify the columns
wlp_schema = lines.map(lambda r: Row(textID=int(r[0]),idseq=int(r[1]),word=r[2],lemma=r[3],pos=r[4]))
wlp = spark.createDataFrame(wlp_schema).repartition(NPARTITION,'lemma').persist() #this partition propagates and so does not need to be repeated


############## Cleaning stage ##############
#pos
pos_remove = ['.',',',"\'",'\"','null']
wlp_nopos = wlp.filter(~wlp['pos'].isin(pos_remove)).filter(~wlp['pos'].startswith('m')).filter(~wlp['pos'].startswith('f')).drop('idseq','pos','word')

#stopwords
stopwords = sc.textFile('our_stopwords.txt').collect()
wlp_nostop = wlp_nopos.filter(~wlp['lemma'].isin(stopwords))
lemma_freq = wlp_nostop.groupBy('lemma').count()

#removal
[top] = lemma_freq.approxQuantile('count', [TOP_PERCENT], 0.01)
lemma_tokeep = lemma_freq.filter(lemma_freq['count']<top)

wlp_nostop.registerTempTable('wlp_nostop')
lemma_tokeep.registerTempTable('lemma_tokeep')

query = """
SELECT wlp_nostop.lemma, wlp_nostop.textID
FROM wlp_nostop
INNER JOIN lemma_tokeep ON wlp_nostop.lemma = lemma_tokeep.lemma
"""

wlp_kept = spark.sql(query)
wlp_bytext = wlp_kept.groupBy('textID').agg(collect_list('lemma')).sort('textID').withColumnRenamed('collect_list(lemma)','lemma_list')



############## Saving stage ##############/datasets/now_corpus/corpus/wlp
#saving dataframe
wlp_bytext.write.mode('overwrite').parquet('wlp_bytext.parquet')
