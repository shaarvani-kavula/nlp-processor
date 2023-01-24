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
    
