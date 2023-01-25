# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:05:36 2023

@author: shaar
"""

import pandas as pd
import nltk

from collections import Counter
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()


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


def find_topics(text_col, num_topics=5):
    count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
    doc_term_matrix = count_vect.fit_transform(text_col.values.astype('U'))
    LDA = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    LDA.fit(doc_term_matrix)
    for i,topic in enumerate(LDA.components_):
        print(f'Top 10 words for topic #{i}:')
        print([count_vect.get_feature_names()[i] for i in topic.argsort()[-20:]])
        print('\n')
        

def topic_tagging(df, text_col, topic_word):
     flag_name = f"flag_{topic_word}"
     df.loc[text_col.astype(str).str.contains(topic_word), flag_name] = 'Y' 
     return df
    


def sentiment_analysis(df):
    df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df['title']]
    df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df['title']]
    df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df['title']]
    df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df['title']]
    return df
    
    
if __name__ == "__main__":
    
    filepath = ("C:/Users/shaar/OneDrive/Documents/Reddit-NLP/"
                "postprocessor/one-million-reddit-confessions-sample.csv")
    df = pd.read_csv(filepath)
    text_col = df['title']
    df['lemma'] = tokenize_text(df['title'])
    df['bigrams'] = generate_ngrams(df['lemma'], 2)
    find_topics(df['title'])
    
    
