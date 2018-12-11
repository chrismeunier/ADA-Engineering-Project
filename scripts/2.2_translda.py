#############################################################################################
###################### Script 2, perform lda on transformed dataframes ######################
#############################################################################################

#import findspark
#findspark.init()

from pyspark.sql import *
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

#Initialisation
ALPHA = 1
BETA = 0.35
NTOPICS = [10]
SAMPLE_FRACTION = 0.5
NPARTITION = 30
VOCABSIZE = 5000
wlp_bytext = spark.read.parquet('wlp_bytext.parquet')



############## Transformation stage ##############
#tf
cvmodel = CountVectorizer(inputCol="lemma_list", outputCol="raw_features", minDF=200).fit(wlp_bytext)
result_cv = cvmodel.transform(wlp_bytext).drop('lemma_list') #partition persists

#idf
idfModel = IDF(inputCol="raw_features", outputCol="non_norm_features").fit(result_cv)
result_tfidf = idfModel.transform(result_cv).drop('raw_features')

#normalised (by default euclidean norm)
norm = Normalizer(inputCol="non_norm_features", outputCol="features")
tfidf_norm = norm.transform(result_tfidf).drop('non_norm_features')

voc = cvmodel.vocabulary
print('Number of lemmas in vocabulary: {}'.format(len(voc)))


############## LDA stage ##############
tfidf_sample = tfidf_norm.sample(fraction=SAMPLE_FRACTION)
split = tfidf_sample.randomSplit([0.8,0.2])
train = split[0]
test = split[1]

for k in NTOPICS:
    #training
    alpha_asymmetric = [ALPHA/(kn+1) for kn in range(k)] #, docConcentration=alpha_asymmetric,topicConcentration=BETA
    lda_model = LDA(k=k, maxIter=1000, optimizeDocConcentration=True).fit(train)

    #get topics and word list
    topics = lda_model.describeTopics()
    words_topics = topics.rdd.map(lambda r: r[1]).collect()
    weight_topics = topics.rdd.map(lambda r: r[2]).collect()

    #printing topic list
    print('############## Number of topics {}:\n'.format(k))
    print('logPerplexity: {}'.format(lda_model.logPerplexity(test))) #roughly 8 accoridng to what we saw
    for t in range(len(words_topics)):
        print('Topic {}:'.format(t))
        w_list = [voc[idx] for idx in words_topics[t]]
        for i, w in enumerate(w_list):
            print('%.4f*%s'%(weight_topics[t][i],w))
        print('\n')



############## Saving sage stage ##############
#get topic distribution for all texts
#result_lda = lda_model.transform(tfidf).drop('features')
#result_lda.write.mode('overwrite').parquet('result_lda.parquet')
