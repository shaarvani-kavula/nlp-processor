# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:05:36 2023

@author: shaar
"""

import pandas as pd
import spacy
import numpy as np


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


def build_wordcloud(freq_dict, stop_words):
    pass


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
    
