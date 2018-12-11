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
#WLP_FILE = '../sample_data/wordLem_poS.txt'
NPARTITION = 40
BOTTOM_PERCENT = 0.8
TOP_PERCENT = 0.99



############## Loading stage ##############
#read the text file and split by tabs
wlp_rdd = sc.textFile(WLP_FILE).map(lambda r: r.split('\t'))

#identify the columns and changing rdd->df
wlp_schema = wlp_rdd.map(lambda r: Row(textID=int(r[0]),idseq=int(r[1]),word=r[2],lemma=r[3],pos=r[4]))
wlp = spark.createDataFrame(wlp_schema)

#immediately save and load in parquet because operations afterwards might be a lot more efficient (don't know yet)
wlp.write.mode('overwrite').parquet('wlp.parquet')
wlp = spark.read.parquet('wlp.parquet')


############## Cleaning stage ##############
#pos
pos_remove = ['.',',',"\'",'\"','null']
wlp_nopos = wlp.filter(~wlp['pos'].isin(pos_remove)).filter(~wlp['pos'].startswith('m')).filter(~wlp['pos'].startswith('f')).filter(~wlp['pos'].startswith('np')).drop('idseq','pos','word')

#stopwords
stopwords = sc.textFile('our_stopwords.txt').collect()
wlp_nostop = wlp_nopos.filter(~wlp['lemma'].isin(stopwords))
lemma_freq = wlp_nostop.groupBy('lemma').count()

#most and least frequent
[bottom, top] = lemma_freq.approxQuantile('count', [BOTTOM_PERCENT, TOP_PERCENT], 0.01)
lemma_tokeep = lemma_freq.filter(lemma_freq['count']<top).filter(lemma_freq['count']>bottom)
c = lemma_tokeep.count()
print('Number of lemmas left: %d'%c)
print('Percentage of lemmas left: %f'%(c/lemma_freq.count()*100))

wlp_nostop.registerTempTable('wlp_nostop')
lemma_tokeep.registerTempTable('lemma_tokeep')

query = """
SELECT wlp_nostop.lemma, wlp_nostop.textID
FROM wlp_nostop
INNER JOIN lemma_tokeep ON wlp_nostop.lemma = lemma_tokeep.lemma
"""

wlp_kept = spark.sql(query)
wlp_bytext = wlp_kept.groupBy('textID').agg(collect_list('lemma')).sort('textID').withColumnRenamed('collect_list(lemma)','lemma_list').repartition(NPARTITION,'lemma_list').persist()



############## Saving stage ##############/datasets/now_corpus/corpus/wlp
#saving dataframe
wlp_bytext.write.mode('overwrite').parquet('wlp_bytext.parquet')
