#########################################################################################################
###################### Script 3, selection fo the informations we want to download ######################
#########################################################################################################

#import findspark
#findspark.init()

from pyspark.sql import *
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.sql.types import DateType

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

#Initialisation
SOURCES_FILE = '/datasets/now_corpus/corpus/source/now_sources_pt*.txt'

result_lda = spark.read.parquet('result_lda.parquet')
voc = sc.textFile('voc.txt').collect()

NTOPICS = 10



############## Sources stage ##############
#read the source data and split by tabs
sources_rdd = sc.textFile(SOURCES_FILE).zipWithIndex().filter(lambda r: r[1] > 2).keys().map(lambda r: r.split('\t'))

#create schema and change data type for date
sources_schema = sources_rdd.map(lambda r: Row(textID=int(r[0]),nwords=int(r[1]),date=r[2],country=r[3],website=r[4],url=r[5],title=r[6],))
sources = spark.createDataFrame(sources_schema)

#topic_source_bytext = sources.join(result_lda,'textID','inner').drop('url','web','website','title','nwords')

############## Selection stage ##############
#udf to change list of topic distribution into multiple columns
def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    return udf(to_array_, ArrayType(DoubleType()))(col)

#making df for topics by countries
distribution = (result_lda.withColumn("topic", to_array(col("topicDistribution"))).select(["textID"] + [col("topic")[i] for i in range(NTOPICS)]))
country = sources.drop("date","nwords","title","url","website")
country = country.select(col("country"), col("textID").alias("c_textID"))
country_dist = distribution.join(country, distribution.textID == country.c_textID).drop("c_textID")
avg_countryTopics = country_dist.sort("textID").groupby("country").mean().drop("avg(textID)")

#making df for topics by dates
dates = sources.drop("country","nwords","title","url","website")
dates = dates.select(col("date"), col("textID").alias("d_textID"))
date_dist = distribution.join(dates, distribution.textID == dates.d_textID).drop("d_textID")
avg_dateTopics = date_dist.sort("textID").groupBy(year("date"),month("date")).mean()
avg_dateTopics = avg_dateTopics.drop('avg(textID)')



############## Saving stage ##############
#save topic by countries
avg_countryTopics.write.mode('overwrite').parquet('avg_countryTopics.parquet')

#save topic by countries
avg_dateTopics.write.mode('overwrite').parquet('avg_dateTopics.parquet')
