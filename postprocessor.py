# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:05:36 2023

@author: shaar
"""

import pandas as pd
import spacy
import numpy as np
import nltk

from collections import Counter
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from matplotlib import pyplot as plt

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()



def get_unstructured_text(text_col):
    pass


def tokenize_text(text_col):
    df['data'] = text_col
    df['data'] = df['data'].str.lower()
    df['data'] = df['data'].str.replace('[^\w\s]','')
    df['data'] = df['data'].str.replace('\d+', '')
    df['data'] = df['data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['data'] = df['data'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x.split()])
    return df['data']
    


def build_wordcloud(lemma_column, stop_words=None):
    text = WordCloud().generate(lemma_column.to_string())
    plt.imshow(text)
    plt.axis("off")
    plt.show()
    
    
def generate_ngrams(text_col, n=2):
    bigrams_col = text_col.apply(lambda row: list(nltk.ngrams(row, n)))   
    bigrams = bigrams_col.tolist()
    bigrams = list(chain(*bigrams))
    bigrams = [(x.lower(), y.lower()) for x,y in bigrams]
    bigram_counts = Counter(bigrams)
    print(bigram_counts.most_common(10))
    return bigrams_col


def find_topics():
    pass
        

def topic_tagging():
    pass


def sentiment_analysis():
    pass

if __name__ == "__main__":
    
    filepath = ("C:/Users/shaar/OneDrive/Documents/Reddit-NLP/"
                "postprocessor/one-million-reddit-confessions-sample.csv")
    df = pd.read_csv(filepath)
    df['lemma'] = tokenize_text(df['title'])
    df['bigrams'] = generate_ngrams(df['lemma'], 2)
    
    
    
