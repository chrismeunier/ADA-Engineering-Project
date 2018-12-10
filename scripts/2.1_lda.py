#############################################################################################
###################### Script 2, perform lda on transformed dataframes ######################
#############################################################################################

#import findspark
#findspark.init()

from pyspark.sql import *
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.sql.types import DateType

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

#Initialisation
ALPHA = 1
BETA = 0.35
NTOPICS = [5,10,15,20,25]
SAMPLE_FRACTION = 0.1

#read tfidf matrix
tfidf = spark.read.parquet('tfidf_all.parquet')
#read vocabulary
voc = sc.textFile('voc.txt').collect()

'''voc = []
with open('../voc.txt','r') as txt:
    for line in txt:
        voc.append(line.strip())'''


############## LDA stage ##############
tfidf_sample = tfidf.sample(fraction=SAMPLE_FRACTION)
split = tfidf_sample.randomSplit([0.8,0.2])
train = split[0]
test = split[1]
n = test.count()

for k in NTOPICS:
    #training
    alpha_asymmetric = [ALPHA/(kn+1) for kn in range(k)] #, docConcentration=alpha_asymmetric,topicConcentration=BETA
    lda_model = LDA(k=k, maxIter=100, docConcentration=alpha_asymmetric,topicConcentration=BETA).setFeaturesCol('non_norm_features').fit(train)

    #get topics and word list
    topics = lda_model.describeTopics()
    words_topics = topics.rdd.map(lambda r: r[1]).collect()
    weight_topics = topics.rdd.map(lambda r: r[2]).collect()

    #printing topic list
    print('############## Number of topics {}:\n'.format(k))
    print('logPerplexity: {}'.format(lda_model.logPerplexity(test)/n)) #roughly 0.014 accoridng to what we saw
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
