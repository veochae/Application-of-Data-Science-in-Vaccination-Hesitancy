# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:21:58 2021

@author: dchae2
"""


#1. Packages Utilized
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

#2. Data Import
data = pd.read_csv('C:/Users/dchae2/Desktop/Youtube Analysis/Sentiment Analysis/Left_Youtube_finalized.csv')
    #this is the left-sided media bias youtube videos (script, title, likes, dislikes, publisher)
data = data.drop([42])
    #the last observation did not contain a script due to Youtube Malfunction, thus dropped
data_1 = pd.read_csv('C:/Users/dchae2/Desktop/Youtube Analysis/Sentiment Analysis/Right_Youtube_finalized.csv')
    #this is the right-sided bias youtube videos (script, title, likes, dislikes, publisher)
bias = pd.read_csv('C:/Users/dchae2/Desktop/Youtube Analysis/Sentiment Analysis/Media Bias.csv')
    #this is the media bias reference dataset acquired through Ad Fontes Media
data = data.append(data_1)
data = data.merge(bias, on = "uploader", how = "left")


#data cleaning
data['scripts'] = data['scripts'].str.replace('WEBVTT', '')
data['scripts'] = data['scripts'].str.replace(' Kind: captions', '')
data['scripts'] = data['scripts'].str.replace(' Language: en', '')
    #all scripts contained heading, thus deletion of heading took place

#3. Data Cleansing for NLP Usage --- Referenced from Abdullah Zahid's Sentiment Analysis Script
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

##LDA 


import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
data = data.text.values.tolist()
data_words = list(sent_to_words(data))
data_words = remove_stopwords(data_words)

flat_list = []
for sublist in data_words:
    for item in sublist:
        flat_list.append(item)


from collections import Counter
list1= flat_list
counts = Counter(list1)
print(counts)