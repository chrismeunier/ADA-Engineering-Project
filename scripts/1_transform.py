######################################################################################
###################### Script 1, taking the processed files and ######################
###################### applying tfidf normalised transformation ######################
######################################################################################

#import findspark
#findspark.init()

from pyspark.sql import *
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

#Initialisation
NPARTITION = 40
VOCABSIZE = 10000
wlp_bytext = spark.read.parquet('wlp_bytext.parquet')


############## Transformation stage ##############
#tf
cvmodel = CountVectorizer(inputCol="lemma_list", outputCol="raw_features", vocabSize=VOCABSIZE).fit(wlp_bytext)
result_cv = cvmodel.transform(wlp_bytext).drop('lemma_list').repartition(NPARTITION,'textID').persist() #partition persists

#idf
idfModel = IDF(inputCol="raw_features", outputCol="non_norm_features").fit(result_cv)
result_tfidf = idfModel.transform(result_cv).drop('raw_features')

#normalised (by default euclidean norm) NOT DONE FOR NOW BECAUSE ACTUALLY MAKES IT WORSE !
#norm = Normalizer(inputCol="non_norm_features", outputCol="features")
#tfidf_norm = norm.transform(result_tfidf).drop('non_norm_features')



############## Saving stage ##############/datasets/now_corpus/corpus/wlp
#saving dataframe
result_tfidf.write.mode('overwrite').parquet('tfidf_all.parquet')

#saving vocabulary from CountVectorizer
voc = cvmodel.vocabulary
sc.parallelize(cvmodel.vocabulary).saveAsTextFile('voc.txt')
'''with open('voc.txt','w') as txt:
    for i,word in enumerate(voc):
        txt.write(word.encode(encoding='UTF-8')+'\n')
    print('Saved {} words'.format(i+1))'''
