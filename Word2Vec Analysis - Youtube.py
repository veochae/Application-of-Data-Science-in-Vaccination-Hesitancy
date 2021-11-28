# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:05:11 2021

@author: dchae2
"""

####WORD2VEC ANALYSIS ON YOUTUBE DATASET

#Referenced from:
    #https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne/comments
    #https://www.kaggle.com/varun08/sentiment-analysis-using-word2vec/data?select=labeledTrainData.tsv

#####################################################################################################################
######################## THIS PORTION IS REFERENCED FROM THE SENTIMENT ANALYSIS #####################################
#####################################################################################################################


import numpy as np # linear algebraimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from wordcloud import WordCloud,STOPWORDS

plt.rc('figure',figsize=(17,13))
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import vaderSentiment
import twython

#loading data
data = pd.read_csv('C:/Users/dchae2/Desktop/Youtube Analysis/Sentiment Analysis/Youtube_finalized.csv')
data = data.drop([42])
data_1 = pd.read_csv('C:/Users/dchae2/Desktop/Youtube Analysis/Sentiment Analysis/Right_Youtube_finalized.csv')
bias = pd.read_csv('C:/Users/dchae2/Desktop/Youtube Analysis/Sentiment Analysis/Media Bias.csv')
data = data.append(data_1)
data = data.merge(bias, on = "uploader", how = "left")


#data cleaning
data['scripts'] = data['scripts'].str.replace('WEBVTT', '')
data['scripts'] = data['scripts'].str.replace(' Kind: captions', '')
data['scripts'] = data['scripts'].str.replace(' Language: en', '')



def clean(text):
     text = re.sub('https?://\S+|www\.\S+', '', text)
     text = re.sub(r'\s+', ' ', text, flags=re.I)
     text = re.sub('\[.*?\]', '', text)
     text = re.sub('\n', '', text)
     text = re.sub('\w*\d\w*', '', text)
     text = re.sub('<.*?>+', '', text)
     return text

data['scripts'] = data['scripts'].apply(lambda x:clean(x))


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
scores=[]
for i in range(len(data['scripts'])):
     score = analyser.polarity_scores(data['scripts'][i])
     score=score['compound']
     scores.append(score)  
     
sentiment=[]
for i in scores:
     if i>=0.05:
         sentiment.append('Positive')
     elif i<=(-0.05):
         sentiment.append('Negative')
     else:
         sentiment.append('Neutral')

data['sentiment']=pd.Series(np.array(sentiment))


def clean_text(text):
    
    text = str(text).lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    return text

data['scripts'] = data['scripts'].apply(lambda x:clean_text(x))
data['scripts']


##Performing Stemming and Lemmatization

data_2=pd.DataFrame()
data_2['text']=data['scripts']


def tokenization(text):
     text = re.split('\W+', text)
     return text

data_2['tokenized'] = data_2['text'].apply(lambda x: tokenization(x.lower()))
nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
     text = [word for word in text if word not in stopword]
     return text

data_2['No_stopwords'] = data_2['tokenized'].apply(lambda x: remove_stopwords(x))

ps = nltk.PorterStemmer()

def stemming1(text):
     text = [ps.stem(word) for word in text]
     return text 

data_2['stemmed_porter'] = data_2['No_stopwords'].apply(lambda x: stemming1(x))


from nltk.stem.snowball import SnowballStemmer
s_stemmer = SnowballStemmer(language='english')
def stemming2(text):
     text = [s_stemmer.stem(word) for word in text]
     return text
data_2['stemmed_snowball'] = data_2['No_stopwords'].apply(lambda x: stemming2(x))
wn = nltk.WordNetLemmatizer()
def lemmatizer(text):
     text = [wn.lemmatize(word) for word in text]
     return text

nltk.download('wordnet')

data_2['lemmatized'] = data_2['No_stopwords'].apply(lambda x: lemmatizer(x))
data_2.head()

data['text'] = data_2['lemmatized']
data.head()

#####################################################################################################################
######################## END OF THE REFERENCED PORTION - STARTING WORD2VEC PROCESS ##################################
#####################################################################################################################


import logging
import gensim
from gensim.models import word2vec
from gensim.models import Word2Vec

# Setting the WORD2VEC Model Parameters
num_features = 300  # Word vector dimensionality
min_word_count = 10 # Minimum word count
num_workers = 10     # Number of parallel threads
context = 10        # Context window size
downsampling = 1e-3 # (0.001) Downsample setting for frequent words

#Model Function
model = gensim.models.Word2Vec(full,\
                          workers=num_workers,\
                          vector_size = 145,\ #The vector_size shows the underlying dimension size 
                          min_count=min_word_count,\
                          window=context)

# To make the model memory efficient
model.init_sims(replace=True)

# Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "Sentiment Based Youtube WORD2VEC Model"
model.save(model_name)

#testing to see if the model properly works
model.wv.most_similar('misinformation')

#collecting all the words that are within the model dictionary
all_words = pd.DataFrame(model.wv.index_to_key)
all_words.columns = {'words'}

#creating a replica for sentiment measures
aa = all_words

#measuring the sentiment of the words within the model dictionary
scores=[]
for i in range(len(aa['words'])):
     score = analyser.polarity_scores(aa['words'][i])
     score=score['compound']
     scores.append(score)  
     
sentiment=[]
for i in scores:
     if i>=0.05:
         sentiment.append('Positive')
     elif i<=(-0.05):
         sentiment.append('Negative')
     else:
         sentiment.append('Neutral')

aa['sentiment']=pd.Series(np.array(sentiment))

#subsetting only the negative sentiment words
#tokenized and then turned into list for visualization
bb = aa[aa['sentiment']=="Negative"]
bb['words'] = bb['words'].apply(lambda x: tokenization(x.lower()))
cc = bb['words']
cc = cc.tolist()

#subsetting only the positive sentiment words
#tokenized and then turned into list for visualization
dd = aa[aa['sentiment']=="Positive"]
dd['words'] = dd['words'].apply(lambda x: tokenization(x.lower()))
ee = dd['words']
ee = ee.tolist()

#start of the visualization

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

z = model.wv.__getitem__(["vaccine"])
def tsnescatterplot(model, word, lists, lists_2):

    arrays = np.empty((0, 145), dtype='f')  # the array has to match the vector size in line 164
    word_labels = [word]
    color_list  = ['red'] #coloring the main word as red which is "vaccine" for this analysis purpose

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of negative words
    close_words = lists
    
    # adds the vector assigned by the model to the negative words
    # add color blue to the negative sentiment words
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # gets list of positive words
    close_words = lists_2
    
    # adds the vector assigned by the model to the positive words
    # add color green to the positive sentiment words
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
 
    # Reduces the dimensionality - this variable varies by amount of words utilized 
    reduc = PCA(n_components=144).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(20, 20)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))

    
#negative + positive word vector distribution centered around the main word "vaccine"
tsnescatterplot(model, 'vaccine', cc, ee)


##################
## This section covers the Top 30 Positive/Negative Words and vector distance 
## for left vs. right media bias


