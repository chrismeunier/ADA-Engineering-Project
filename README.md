# Biases and sentimentality in online newspapers

# Abstract

In the current media context, access to objective information is becoming critical. People want to build strong opinions on various subjects that have direct impact on their lives. With the rise of misinformation and biased untraceable sources flooding our social media feed having a tool to understand the information that is shown to us could be very useful to build an informed opinion on a certain subject.

The goal of our project is to use data analysis techniques on a large news dataset (NOW database) to help people to get a better understand the information they are receiving.

By analysing the articles from various newspapers coming from different locations across time, we can create a topic model of our data using topic modeling techniques like Latent Dirichlet Allocation. This will give the articles dimensionality. 

From this topic model we want to track the evolution of the general sentiment relative to each of these topics across time and location. This will give people a way to understand the past and current sentiment relative to a certain topic they might be interested in. 

Also tracking the changes in the main topic of each newspaper across time can tell a lot about the biases of a certain journal relative to a certain topic. 

Finally we want to create a classification of journals so we can classify any new article into a relative topics or political orientation. 


# Research questions
*A list of research questions you would like to address during the project. *

# Dataset
We will use the *News On the Web* dataset (NOW) as described on  [corpusdata]{https://www.corpusdata.org/intro.asp}. It contains articles from online newspapers coming from 20 english-speaking countries (USA, GB, Australia, India, etc) ranging from 2010 to today. This data represents more than 6 billion words stored in many many text files, it is accessible on the cluster and we will extract some parts of it to test the implementation of our methods locally before running it on the whole dataset on the cluster.
These text files contain a lexicon linking words together we can use to read words as other words (e.g. a conjugated verb “is” could be read as “be”). We have a list of all articles where we find their unique (we hope) id, their word counts, the country and name of newspaper it was published in, and an article headline. 
The articles themselves are stored in files classified according to the year, month and country of publication. So all articles (starting with their id) of a given month published in one country are found in the same file, this means we will need to process them accordingly to extract information for each article separately.

*List the dataset(s) you want to use, and some ideas on how do you expect to get, manage, process and enrich it/them. Show us you've read the docs and some examples, and you've a clear idea on what to expect. Discuss data size and format if relevant.*

# A list of internal milestones up until project milestone 2
- Understand the complete structure of the dataset and how to clean it.
- Clean the data base according to what is necessary.
- Understand how to sample our data to create a balanced topic model.
- ....
# Questions for TAs

- Knowing that different countries and news sources have largely different number of articles how should we perform our topic modeling without biasing our data towards a certain source or country?

