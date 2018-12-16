# ADA 2018 project - Topic analysis of online newspapers
Jean-Marc Bejjani - Luca Montanelli - Christophe Muller

## Notebook & report
The notebook containing the processes of this project is "Our process on the sample data.ipynb", indicatively you will need to import `numpy`, `pyplot`, `pandas`, `pyspark`, `findspark`, `scikit` and `datetime` if you wish to run it.

The report is obviously "Topic_analysis_of_online_newspapers.tex"

## Abstract
In the current media context, access to objective information is becoming more and more critical. People want to build strong opinions on various subjects that have direct impact on their lives.
The goal of our project is to use data analysis techniques on a large newspapers dataset ([NOW](https://corpus.byu.edu/now/help/tour.asp) database) to help people get a better understanding of the information they are receiving.

By analysing the articles from various newspapers coming from different locations across time, we will create a topic model of our data using topic modeling techniques like Latent Dirichlet Allocation.
In our implementation we will use the `pyspark` library to train LDA models directly on the cluster.
From this topic model we want to track the evolution of these topics across time and location. This will give people an insight on the past and current importance of a certain topic they might be interested in.


## Research questions
  - Have some big events changed the way newspapers perceive topics? We will try to find an event with a large impact on a specific topic and compare the sentiment of articles related to this topic before and after the event.
  - Are countries drastically different in the way their newspapers treat events?
  - Have the principal topics (or our perception of them) changed over the years? Can this tell us a change of mentality? Or where we are headed?

## Dataset
We will use the *News On the Web* dataset (NOW) as described on [corpusdata](https://www.corpusdata.org/intro.asp). It contains articles from online newspapers coming from 20 english-speaking countries (USA, GB, Australia, India, etc) ranging from 2010 to today. This data represents more than 6 billion words stored in many many text files, it is accessible on the cluster and we will extract some parts of it to test the implementation of our methods locally before running it on the whole dataset on the cluster.

These text files contain a lexicon listing all words used in the articles and linking them with "lemmas" to simplify reading (e.g. a conjugated verb “is” has a lemma “be” to simplify word processing). We also have a list of all articles where we find their unique id, their word counts, the country and name of newspaper it was published in, and an article headline. 

The articles themselves are stored in files classified according to the year, month and country of publication. So all articles (starting with their id) of a given month published in one country are found in the same file, this means we will need to process them accordingly to extract information for each article separately. 

All the articles files have a WLP (Word-Lemma-PartOfSpeech) counterpart file containing the ordered list of all words used in these articles. With each word is found its lemma and PoS. These files are the most promising way to treat our data since we can directly work on the lemmas as if it was a pre-processed dataset with more simple and meaningful words.

## Workload repartition:
Jean-Marc: ideation, initial cleaning of the data, implementation of LDA in pyspark, analysis of the results (PCA and clustering), writing results;

Luca: pre-processing/cleaning for LDA, implementation of LDA in pyspark, coding and managing the scripts on the cluster, writing LDA;

Christophe: initial cleaning of the data, initial implementation on the cluster, regrouping final results, writing introduction and conclusion, presentation.
