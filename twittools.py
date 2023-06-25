# 12-2012 @ CEFOS ddhbb


# Natural Language Processing Toolkit v0.1 (MRH NLTK V0.1)
# Written by: Mehdi REZZAG HEBLA
# For PhD. Thesis Defense
# Date: Sep. 08, 2022


import tweepy
import configparser
import re

# !!-------------------------------- UNCOMMENT LINES WHEN MODULE NEEDED --------------------
# import snscrape.modules.twitter as sntwitter
# import pprint
# from datetime import datetime
import pandas as pd
# import numpy as np
# import twittools
# from matplotlib import pyplot as plt
# import datetime
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.cluster import KMeans
# from collections import Counter
# import re
# import emoji
# import regex
# import unicodedata
# from nltk.corpus import stopwords
# import nltk
# import copy
# from sklearn.preprocessing import StandardScaler
# import seaborn as sns
# import spacy
# from sklearn.model_selection import train_test_split
# from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
# from nltk.stem import WordNetLemmatizer
# import string
# from string import punctuation
# import collections

# import en_core_web_sm
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import jaccard_score


"""
Uncomment code below only at the first time of usage:
"""
# nltk.download('wordnet')
# ---------------------------------- TWITTER ACCOUNT MANAGEMENT ----------------------------
cofig = config = configparser.RawConfigParser()
config.read('config.ini')
api_key = config['twitter']['api_key']
api_secret = config['twitter']['api_secret']
bearer_token = config['twitter']['bearer_token']
access_token = config['twitter']['access_token']
access_toke_secret = config['twitter']['access_toke_secret']


def connect_to_account():
    client = tweepy.Client(bearer_token=bearer_token, consumer_key=api_key, consumer_secret=api_secret, \
                           access_token=access_token, access_token_secret=access_toke_secret, wait_on_rate_limit=True)
    return client


client = connect_to_account()

# ---------------------------------- TWITTER DATA PIPELINE TOOLS  ----------------------------
def collect_quotes(tweet_id):
    # This function collects data on quotes regarding a specific tweet
    # it returns a dictionary of dictionaries, each containing: 
    # quoting user id, their username, name(as displayed), id of the tweet, text of tweet 
    # returns up to the 100 first quotes, Twitter API does not allow going beyond the threshold (100)
    try:
        response = client.get_quote_tweets(tweet_id, expansions=["author_id"], max_results=100)
        # returns None object in case of response object empty
        if response.data == None or response.includes == None:
            return None
        else:
            response_dictionary = {u['id']: {'uid' : u['id'],
                            'name': u['name'],
                            'username' : u['username'],
                            'tweet_id': t['id'],
                            'tweet_text': t['text']} for u,t in zip(response.includes['users'], response.data)}
            return response_dictionary
    except Exception as e:
        print(e)
        return None

def collect_liking_users(tweet_id):
    # This function collects data on liking users regarding a specific tweet
    # it returns a dictionary of dictionaries, each containing: 
    # liking user id, liking username, liking user display name 
    # returns up to the 100 first quotes, Twitter API does not allow going beyond the threshold (100)
    try:
        response = client.get_liking_users(tweet_id, max_results=100)
        # returns None object in case of response object empty
        if response.data == None:
            return None
        else:
            response_dictionary = {u['id']: {'uid' : u['id'],
                            'name': u['name'],
                            'username' : u['username']} for u in response.data}
            return response_dictionary
    except Exception as e:
        print(e)
        return None
# collects and stores dictionaries of quotes and liking users
def collect_quotes_likes(dataframe):
    df2 = dataframe.copy(deep=True)
    for i, row in df2.iterrows():
        quote_dict = dict()
        like_dict = dict()
        # 1. collecting quotes
        # checking if a tweet has been quoted
        if row['n_quotes'] > 0:
            tweet_id = row['tweet_id']
            quote_dict = collect_quotes(tweet_id)
            if quote_dict:
                df2.loc[i, 'tweet_quotes'] = str(quote_dict)
                print(f'Index: {i} - updated quotes entry successfully. Tweet ID: {tweet_id}')
        if row['n_likes'] >= 15:
            tweet_id = row['tweet_id']

            like_con = True
            like_dict = collect_liking_users(tweet_id)
            if like_dict:
                df2.loc[i, 'tweet_likers'] = str(like_dict)
                print(f'Index: {i} - updated likers entry successfully. Tweet ID: {tweet_id}')

# this is a function to extract: hashtags, @mentions, and links
def tweet_parser(tweet):
    # URL pattern for url detection in tweet
    re_expression = '[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-zأ-ي]{2,6}\b([-a-zA-Zأ-ي0-9@:%_\+.~#?&//=]*)'
    # replacing any url with ''
    tweet = re.sub(re_expression, '', tweet)
    # initiating empty list for hashtags
    hashtags = list()
    # initiating empty list for mentions
    mentions = list()
    tweet_to_list = tweet.split(' ')
    for i, word in enumerate(tweet_to_list):
        if (word.startswith('#') or word.endswith('#')):
            hashtags.append(word)
        elif (word.startswith('@') or word.endswith('@')):
            mentions.append(word)
            
    return hashtags if hashtags else None
# takes a string object, removes hashtags    
def remove_hashtags(tweet):
    tweet = tweet.replace('\n', ' ')
    tweet_to_list = tweet.split(' ')
    for i, word in enumerate(tweet_to_list):
        if (word.startswith('#') or word.endswith('#')):
            tweet_to_list.pop(i)
        elif (word.startswith('@') or word.endswith('@')):
            tweet_to_list.pop(i)
    return ' '.join(tweet_to_list) if tweet_to_list else None

# Function to parse tweets, returns raw text -without emojis, and a list of emojis extracted, see example in the comment below
def emoji_parser(text):
    emoji_list = []
    # \X matches any unicode grapheme, e.g., ©
    # Note: here "regex" library is used which is external; and not "re" library
    # data captures all graphemes, but not all graphemes are emojis
    data = regex.findall(r'\X', text)
    # data needs to be cleaned using all recognized emojis filter list (.EMOJI_DATA)
    for word in data:
        # if there is a match, the grapheme is appended to a list of emojis
        if any(char in emoji.unicode_codes.EMOJI_DATA for char in word):
            emoji_list.append(word)
    if emoji_list:
        for e in emoji_list:
            text = text.replace(e, '')
    flags = [flag for flag in twittools.flags if flag in text]
    emoji_list = flags + emoji_list
    # this is a parser that removes any unicode characters pointing to flags e.g., '🇩🇿'.
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    # returns emojiless string and a list of emojis, e.g., ('lorem ipsum dolor!', ['👍', '❤️'])
    return text, emoji_list

# function that removes emojis, hashtags, and mentions -uses the two functions above.
def clean_tweets(df):
    for i, row in df.iterrows():
        tweet = row['content']
        print(f'Parsing tweet @ index: {i}...')
        hashtags = tweet_parser(tweet)
        print(f'Hashtags: {hashtags}')
        raw_emojis = emoji_parser(tweet)
        print(f'Raw tweet: {raw_emojis[0]}')
        print(f'Emojis: {raw_emojis[1]}')
        df.at[i, 'content'] = raw_emojis[0]
        if hashtags:
            df.at[i, 'hashtags'] = ' '.join(hashtags)
        if raw_emojis[1]:
            df.at[i, 'emojis'] = ' '.join(raw_emojis[1])

# Function to generate word clouds out of a dataframe
def gen_w_cloud(df, title=None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopw,
        max_words=50,
        max_font_size=40,
        scale=3,
        random_state=1
    ).generate(df.str.cat(sep=' '))
    fig = plt.figure(1, figsize=(20,20))
    plt.axis('off')
    if title:
        fig.suptitle(titletle, fontsize=20)
        fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud)
    plt.show()

# ---------------------------------- TWEET-SPECIFIC UTILITIES ----------------------------
# a custom list of all the emoji flags in unicode 
flags = ['🇦🇩', '🇦🇪', '🇦🇫', '🇦🇬', '🇦🇮', '🇦🇱', '🇦🇲', '🇦🇴', '🇦🇶', '🇦🇷', '🇦🇸', '🇦🇹', '🇦🇺', '🇦🇼', '🇦🇽', '🇦🇿', '🇧🇦', '🇧🇧', '🇧🇩', '🇧🇪', '🇧🇫', '🇧🇬', '🇧🇭', '🇧🇮', '🇧🇯', '🇧🇱', '🇧🇲', '🇧🇳', '🇧🇴', '🇧🇶', '🇧🇷', '🇧🇸', '🇧🇹', '🇧🇻', '🇧🇼', '🇧🇾', '🇧🇿', '🇨🇦', '🇨🇨', '🇨🇩', '🇨🇫', '🇨🇬', '🇨🇭', '🇨🇮', '🇨🇰', '🇨🇱', '🇨🇲', '🇨🇳', '🇨🇴', '🇨🇷', '🇨🇺', '🇨🇻', '🇨🇼', '🇨🇽', '🇨🇾', '🇨🇿', '🇩🇪', '🇩🇯', '🇩🇰', '🇩🇲', '🇩🇴', '🇩🇿', '🇪🇨', '🇪🇪', '🇪🇬', '🇪🇭', '🇪🇷', '🇪🇸', '🇪🇹', '🇫🇮', '🇫🇯', '🇫🇰', '🇫🇲', '🇫🇴', '🇫🇷', '🇬🇦', '🇬🇧', '🇬🇩', '🇬🇪', '🇬🇫', '🇬🇬', '🇬🇭', '🇬🇮', '🇬🇱', '🇬🇲', '🇬🇳', '🇬🇵', '🇬🇶', '🇬🇷', '🇬🇸', '🇬🇹', '🇬🇺', '🇬🇼', '🇬🇾', '🇭🇰', '🇭🇲', '🇭🇳', '🇭🇷', '🇭🇹', '🇭🇺', '🇮🇩', '🇮🇪', '🇮🇱', '🇮🇲', '🇮🇳', '🇮🇴', '🇮🇶', '🇮🇷', '🇮🇸', '🇮🇹', '🇯🇪', '🇯🇲', '🇯🇴', '🇯🇵', '🇰🇪', '🇰🇬', '🇰🇭', '🇰🇮', '🇰🇲', '🇰🇳', '🇰🇵', '🇰🇷', '🇰🇼', '🇰🇾', '🇰🇿', '🇱🇦', '🇱🇧', '🇱🇨', '🇱🇮', '🇱🇰', '🇱🇷', '🇱🇸', '🇱🇹', '🇱🇺', '🇱🇻', '🇱🇾', '🇲🇦', '🇲🇨', '🇲🇩', '🇲🇪', '🇲🇫', '🇲🇬', '🇲🇭', '🇲🇰', '🇲🇱', '🇲🇲', '🇲🇳', '🇲🇴', '🇲🇵', '🇲🇶', '🇲🇷', '🇲🇸', '🇲🇹', '🇲🇺', '🇲🇻', '🇲🇼', '🇲🇽', '🇲🇾', '🇲🇿', '🇳🇦', '🇳🇨', '🇳🇪', '🇳🇫', '🇳🇬', '🇳🇮', '🇳🇱', '🇳🇴', '🇳🇵', '🇳🇷', '🇳🇺', '🇳🇿', '🇴🇲', '🇵🇦', '🇵🇪', '🇵🇫', '🇵🇬', '🇵🇭', '🇵🇰', '🇵🇱', '🇵🇲', '🇵🇳', '🇵🇷', '🇵🇸', '🇵🇹', '🇵🇼', '🇵🇾', '🇶🇦', '🇷🇪', '🇷🇴', '🇷🇸', '🇷🇺', '🇷🇼', '🇸🇦', '🇸🇧', '🇸🇨', '🇸🇩', '🇸🇪', '🇸🇬', '🇸🇭', '🇸🇮', '🇸🇯', '🇸🇰', '🇸🇱', '🇸🇲', '🇸🇳', '🇸🇴', '🇸🇷', '🇸🇸', '🇸🇹', '🇸🇻', '🇸🇽', '🇸🇾', '🇸🇿', '🇹🇨', '🇹🇩', '🇹🇫', '🇹🇬', '🇹🇭', '🇹🇯', '🇹🇰', '🇹🇱', '🇹🇲', '🇹🇳', '🇹🇴', '🇹🇷', '🇹🇹', '🇹🇻', '🇹🇼', '🇹🇿', '🇺🇦', '🇺🇬', '🇺🇲', '🇺🇸', '🇺🇾', '🇺🇿', '🇻🇦', '🇻🇨', '🇻🇪', '🇻🇬', '🇻🇮', '🇻🇳', '🇻🇺', '🇼🇫', '🇼🇸', '🇽🇰', '🇾🇪', '🇾🇹', '🇿🇦', '🇿🇲']


# a function to extract urls shared in a tweet
def get_listed_url(tweet_object):
    if tweet_object.links == None:
        return None
    elif len(tweet_object.links) == 1:
        return tweet_object.links[0].url
    elif len(tweet_object.links) > 1:
        return [u.url for u in tweet_object.links]
    else:
        return None

# This function takes a dataframe, an empty dict with poster entries and returns the interactors 
# with these posters
def interactors_parser(dataframe, pattern, master_dict):
    updated_master_dict = copy.deepcopy(master_dict)
    for r in dataframe.iterrows():
        interactors = []
        poster = r[1]['username']
#         print(f"poster: {poster}")
        if r[1]['quoted'] > 0:
            rslt = re.findall(pattern, r[1]['tweet_quotes'], re.I)
            if rslt:    
                interactors.extend(rslt)
#                 print(f"****interactors after quoted: {interactors}")
        if r[1]['liked'] > 0:
            rslt = re.findall(pattern, r[1]['tweet_likers'], re.I)
            if rslt:
                interactors.extend(rslt)
#                 print(f"****interactors after liked: {interactors}")
        if interactors:
            # c_e contains values only for current iteration
            c_e = {'node': poster, 'edges': interactors, 'size': len(interactors)}
            # indiv dict contains previous values
            indiv_dict = updated_master_dict[poster]
            # updating indiv dict to contain the most recent values
#             print(indiv_dict)
#             print(f"indiv_dict(interactors): {indiv_dict['interactors']}")
#             print(f"c_e(edges): {c_e['edges']}")
            for x in c_e['edges']:
                indiv_dict['interactors'].append(x)
#             indiv_dict['interactors'].append(x for x in c_e['edges'])
            indiv_dict['interactors'] = list(tuple(indiv_dict['interactors']))
#             print(f"indiv_dict(interactors): {(indiv_dict['interactors'])}")
            indiv_dict['size'] = indiv_dict['size'] + c_e['size']
            updated_master_dict.update({poster: indiv_dict})
#             print(f"indiv_dict['size']: {indiv_dict['size']}")
    return updated_master_dict

# the main function that collects the whole database
# receives a query, formulated using Twitter's advanced search form
# link: https://twitter.com/search-advanced
def batch_collector(query):
    # initiating an empty dictionary to be populated by data collected
    # raw data is cleaned and formatted
    # only certain entries are of interest, i.e.: 
    # username, content, date, tweet_id, outlinks in case a user shares a link in their tweet, \
    # number of links shared, number of times tweet has been quoted, \
    # number of times tweet has been liked, the language (though in this example only French was selected) \
    # number of replies to the tweet, list of users quoting tweet, list of users liking tweet.
    tweets_dict = dict()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        # Sets a limit for the maximum tweets to be collected
        if i >= 5000:
            break
        
        tweets_dict.update({i:{
            'username': tweet.user.username,
            'content':tweet.rawContent, 
            'date':tweet.date, 
            'tweet_id':tweet.id, 
            'outlinks': get_listed_url(tweet),
            'n_links': len(tweet.links) if tweet.links != None else 0,
            'url':tweet.url,
            'n_likes':tweet.likeCount,
            'n_quotes': tweet.quoteCount,
            'language': tweet.lang,
            'replies': tweet.replyCount,
            'tweet_quotes': np.nan,
            'tweet_likers': np.nan,
         }})
    return tweets_dict

def normalize_list_of_lists(matrix):
    norm_matrix = []
    for i, l in enumerate(matrix):
        inner_row = [round((e/sum(l))*100, 1) for e in l]
        norm_matrix.append(inner_row)

# Time series plotter
# x (dates); y (frequencies -absolute); x & y_labels (text labels); drange (data range -if not specified includes the whole data range);
# xticks_rotation (the rotation of the x ticks -if not specified rotates text to 45°);
# graph_style (the layout -access whole list @ matplotlib.pyplot.style.available); grid (grid -available by default)
def plot_tweet_frequencies(x, y, xlabel='xlabel', ylabel='ylabel',\
                           drange=[0, -1], xticks_rotation=45, graph_style='fivethirtyeight', grid=True):
    x = x[drange[0]:drange[1]]
    y = y[drange[0]:drange[1]]
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.style.use(graph_style)
    plt.grid(grid)
    plt.xticks(rotation=xticks_rotation)
    plt.yticks(np.arange(start=0, stop=480, step=30))
    plt.plot(x, y)


# ---------------------------------- ARTIFICIAL INTELLIGENCE (AI) MODEL FOR NATURAL LANGUAGE PROCESSING (NLTK) ----------------------------
# lemmatizes french text, e.g., "j'espère qu'on gagnera le match de la finale" becomes "espoir gain match final"
def furnished(tweet):
    final_text = []
    for i_token in w_tokenizer.tokenize(tweet):
       if i_token.lower() not in stop:
           word = nlp(u"%s" %i_token)
           final_text.append(word[0].lemma_.lower())
    return " ".join(final_text)

def lemmatize_text(string_list):
    lemmatized_list = []
    stop = list(stopwords.words('french'))
    stop = stop + ['http', 'https', 'co', 'avoir', 'tout', 'cad']
    stop = set(stop)
    tokenizer = RegexpTokenizer(r'\w+')
    nlp = spacy.load('fr_core_news_md')
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    w_tokenizer = WhitespaceTokenizer()
    for text in string_list:
        lemma_ = furnished(text)
        lemmatized_list.append(lemma_)
    return lemmatized_list



# ---------------------------------- FILE MANAGEMENT ----------------------------
#open a csv file
def pd_open_csv(filepath, index_col=None):
    # returns a pandas dataframe
    return pd.read_csv(filepath, index_col=index_col)

# saves dataframe to .csv file
def save_to_csv(df, path):
    try:
        df.to_csv(path, encoding='utf-8')
        print('Saved successfully.')
    # in case of exception, Exception is printed
    except Exception as e:
        print('Error occured:')
        print(e)

# saves dataframe to .xlsx file
def save_to_excel(df, file_name):
    try:
        # making a heard copy of the original dataframe to protect it from alteration
        df_2 = df.copy()
        # converting timestamp to datetime format
        try:
            df_2['date'] = df_2['date'].apply(lambda a: pd.to_datetime(a).date())
        except Exception as e:
            pass
        # originally saved under title: 'oran_22_xlsx.xlsx'
        writer = pd.ExcelWriter(file_name)
        df_2.to_excel(writer, 'Sheet1')
        writer.save()
        print('Saved successfully.')
    # in case of exception, Exception is printed
    except Exception as e:
        print('Error occured:')
        print(e)

# converts selected column with date strings to datetime object
def convert_to_datetime(df, column_label='date', date_time_format_str='%Y-%m-%d'):
    df[column_label] = pd.to_datetime(df[column_label], format=date_time_format_str)
# ---------------------------------- DATAFRAME CONSOLIDATION TOOLS (FOR HEATMAP GENERATION) ----------------------------
# function that parses a dictionary of data from a source df
# populates the target_df at the specified date
def update_df_time(target_df, source_dict):
    date_ = source_dict['date']
    row_data = [source_dict['n_tweets'], source_dict['n_likes'], source_dict['n_quotes'], source_dict['cluster_0'], source_dict['cluster_1'], source_dict['cluster_2'], source_dict['cluster_3']]
    df_time.loc[df_time['date'] == date_, 'n_tweets':] += row_data
    



# a function to extract data from the main data frame, i.e., source_df
# and populate a consolidated version of it, i.e., target_df
def popul_time_series(source_df, target_df):

    # parsing each row of source_df
    for i, row in source_df.iterrows():
        # convert row to dictionary
        row_dict = dict(row)
        # compact version of the dictionary containing strictly necessary data
        row_dict_c = {
            'date': row['day_day'] + MonthEnd(0),
            'n_tweets': 1,
            'n_likes': row['n_likes'],
            'n_quotes': row['n_quotes'],
            'cluster_0': 1 if row['cluster'] == 0 else 0,
            'cluster_1': 1 if row['cluster'] == 1 else 0,
            'cluster_2': 1 if row['cluster'] == 2 else 0,
            'cluster_3': 1 if row['cluster'] == 3 else 0
            }
        update_df_time(target_df, row_dict_c)


# ---------------------------------- STRING & LIST OF STR MANIPULATION ----------------------------

# creating a function to take a string (username) and returns only the first three (3) letters
# all the usernames returned are in lowercase
def abbreviate_str(username):
    # make sure that the user name is at least 4 letters long
    # otherwise leave it as is
    if len(username) >= 4:
        return username[:3].lower()
    else:
        return username.lower()

def abbreviate_lst(username_list):
    if username_list: # making sure that the list is not empty
        abbreviated_lst = []
        for uname in username_list:
            abbreviated_lst.append(abbreviate_str(uname))
        return abbreviated_lst
    else: # if list empty just return an empty list
        return []


# function to return key for any value
def get_key(val):
    for key, value in my_dict.items():
        if val == value:
            return key
        # returns None if key doesn't exist
        return None
