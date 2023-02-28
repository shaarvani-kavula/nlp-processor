# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 22:58:23 2023

@author: shaar
"""

import praw
import pandas as pd
from creds import client_id, client_secret, user_agent 

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,  user_agent=user_agent)
    
posts = []
ml_subreddit = reddit.subreddit('MachineLearning')
for post in ml_subreddit.hot(limit=10):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
print(posts)

post_id_list = posts['id']
for post_id in post_id_list:
    submission = reddit.submission(post_id)
    for top_level_comment in submission.comments:
        print(top_level_comment.body)
