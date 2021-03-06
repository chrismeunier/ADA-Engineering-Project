#############################################################################################
###################### Script 2, perform lda on transformed dataframes ######################
#############################################################################################

#to run locally, uncomment the next 2 lines
#import findspark
#findspark.init()

from pyspark.sql import *
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

#Initialisation
NTOPICS = [10]
SAMPLE_FRACTION = 0.5
wlp_bytext = spark.read.parquet('wlp_bytext.parquet')



############## Transformation stage ##############
#tf
cvmodel = CountVectorizer(inputCol="lemma_list", outputCol="raw_features", minDF=200).fit(wlp_bytext)
result_cv = cvmodel.transform(wlp_bytext).drop('lemma_list')

#idf
idfModel = IDF(inputCol="raw_features", outputCol="non_norm_features").fit(result_cv)
result_tfidf = idfModel.transform(result_cv).drop('raw_features')

#normalised (by default euclidean norm)
norm = Normalizer(inputCol="non_norm_features", outputCol="features")
tfidf_norm = norm.transform(result_tfidf).drop('non_norm_features')

voc = cvmodel.vocabulary
#saving the transformed matrix before juse in case we want to use it later
tfidf_norm.write.mode('overwrite').parquet('tfidf_norm.parquet')



############## LDA stage ##############
tfidf_sample = tfidf_norm.sample(fraction=SAMPLE_FRACTION)

for k in NTOPICS:
    #training
    lda_model = LDA(k=k, maxIter=1000, docConcentration=asymmetric_alpha, topicConcentration=0.1, optimizeDocConcentration=True).fit(tfidf_sample)

    #get topics and word list
    topics = lda_model.describeTopics()
    words_topics = topics.rdd.map(lambda r: r[1]).collect()
    weight_topics = topics.rdd.map(lambda r: r[2]).collect()

    #printing topic list
    print('############## Number of topics {}:\n'.format(k))
    for t in range(len(words_topics)):
        print('Topic {}:'.format(t))
        w_list = [voc[idx] for idx in words_topics[t]]
        for i, w in enumerate(w_list):
            print('%.4f*%s'%(weight_topics[t][i],w))
        print('\n')

    #saving model
    lda_model.write().overwrite().save('lda_model_k={}'.format(k))


############## Saving sage stage ##############
#get topic distribution for all texts
result_lda = lda_model.transform(tfidf_norm).drop('features')
result_lda.write.mode('overwrite').parquet('result_lda.parquet')
