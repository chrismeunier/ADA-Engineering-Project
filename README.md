# Stuff to add!
Dependencies for the notebook: numpy, pyplot, pandas, pyspark, findspark, scikit, datetime

# Biases and sentimentality in online newspapers
Jean-Marc Bejjani - Luca Montanelli - Christophe Muller

## Abstract
In the current media context, access to objective information is becoming more and more critical. People want to build strong opinions on various subjects that have direct impact on their lives. With the rise of misinformation and biased untraceable sources flooding our social media feed, having a tool to understand the information displayed in front of us could be very useful to build an  more informed opinion on a given subject.
The goal of our project is to use data analysis techniques on a large newspapers dataset ([NOW](https://corpus.byu.edu/now/help/tour.asp) database) to help people get a better understanding of the information they are receiving.

By analysing the articles from various newspapers coming from different locations across time, we will create a topic model of our data using topic modeling techniques like Latent Dirichlet Allocation.
In our implementation we will use the `gensim` library to train LDA models.
From this topic model we want to track the evolution of the general sentiment relative to each of these topics across time and location. This will give people a way to understand the past and current sentiment relative to a certain topic they might be interested in. Here by sentiment we mean how much the article has a positive/optimistic view on the topic or a negative/pessimistic point of view.

Our main goal is to categorize the newspapers as positive, neutral or negative regarding a given previously identified topic. This will be done firstly inside the countries of the dataset and then internationally since some topics and most events are specific to countries.

Also tracking the changes in the main topic of each newspaper across time could tell a lot about the biases of a certain newspaper relatively to a certain topic. 

Finally we would like to create a classification method of articles so we can classify any new article as being part of an identified topic and have an idea of its general sentimant.

## Research questions
  - Are newspapers optimistic or pessimistic when talking about certain subjects?
  - Can we succesfully classify articles to direct the readers where to go? So that if you don't like how some newspaper is always overly dramatic and negative over some subject, can you find an other newspaper to read for a more peaceful and positive report?
  - Have some big events changed the way newspapers perceive topics? We will try to find an event with a large impact on a specific topic and compare the sentiment of articles related to this topic before and after the event.
  - Are countries drastically different in the way their newspapers treat events? For an event of international impact do we find clear differences between the perception of this subject between countries? Does it tell us anything about alliances or the general national political landscape of a given country?
  - Have the principal topics (or our perception of them) changed over the years? Can this tell us a change of mentality? Or where we are headed?

## Dataset
We will use the *News On the Web* dataset (NOW) as described on [corpusdata](https://www.corpusdata.org/intro.asp). It contains articles from online newspapers coming from 20 english-speaking countries (USA, GB, Australia, India, etc) ranging from 2010 to today. This data represents more than 6 billion words stored in many many text files, it is accessible on the cluster and we will extract some parts of it to test the implementation of our methods locally before running it on the whole dataset on the cluster.

These text files contain a lexicon listing all words used in the articles and linking them with "lemmas" to simplify reading (e.g. a conjugated verb “is” has a lemma “be” to simplify word processing). We also have a list of all articles where we find their unique id, their word counts, the country and name of newspaper it was published in, and an article headline. 

The articles themselves are stored in files classified according to the year, month and country of publication. So all articles (starting with their id) of a given month published in one country are found in the same file, this means we will need to process them accordingly to extract information for each article separately. 

All the articles files have a WLP (Word-Lemma-PartOfSpeech) counterpart file containing the ordered list of all words used in these articles. With each word is found its lemma and PoS. These files are the most promising way to treat our data since we can directly work on the lemmas as if it was a pre-processed dataset with more simple and meaningful words.

## A list of internal milestones up until project milestone 2
- Understand the complete structure of the dataset and how to clean it.
- Clean the data base according to what is necessary.
- Understand how to sample our data to create a balanced topic model.

