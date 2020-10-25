#!/usr/bin/env python
# coding: utf-8

# ## Construct a dataframe with features
# The programmes in this file aim at constructing a data frame with all kinds of feature for further processing. The dataframe should contain user IDs, ground truth labels, the content of their tweets, and all the features below:
# 
# **Lexical**:
# - lexical diversity
# 
# **Semantic**
# - count of named entities
# - count of question marks
# - count of exclamation marks
# 
# **Syntactic**
# - number of exclamation mark
# - number of question mark
# 
# **Sentiment**
# - mean of sentiment values
# - standard deviation of sentiment values
# - mean of subjectivity scores
# - standard deviation of subjectivity scores
# 
# **User Behavior**
# - cross-post (#URL# tag)
# - retweet others (RT tag)
# - mentioning others (#USER# tag)
# - Amount of Hashtag

# In[3]:


from xml.etree import ElementTree as ET
import os
import glob
import numpy as np
import pandas as pd
from collections import OrderedDict
import spacy
from lexical_diversity import lex_div as ld
import re
from textblob import TextBlob


# ### Generating a basic dataframe that contains IDs and the correspondent labels and contents.

# In[4]:


# Path and construct a file list
DIREC = "pan20-author-profiling-training-2020-02-23/"
LANG  = "en/"
file_list = os.listdir(DIREC + LANG)

for i in file_list:
    if i[-3:] != "xml":
        file_list.remove(i)

file_list = sorted(file_list)


# In[5]:


# Get ground truth, append into the dataframe
GT = DIREC + LANG + "truth.txt"
true_values = OrderedDict()
f = open(GT)

for line in f:
    linev = line.strip().split(":::")
    true_values[linev[0]] = linev[1]
    
f.close()

df = pd.DataFrame(sorted(true_values.items()))
df = df.rename({0:"ID", 1:"label"}, axis = 1)
df["label"] = df["label"].astype("int")
# true_values


# In[6]:


def get_representation_tweets(FILE):
    parsedtree = ET.parse(FILE)
    documents = parsedtree.iter("document")
    
    texts = []
    for doc in documents:
        texts.append(doc.text)
        
    lengths = [len(text) for text in texts]
    
#    return (np.mean(lengths), np.std(lengths))
    return (texts)


# In[7]:


# append each content into DF
x = []
for i in range(len(file_list)):
    ind = file_list[i]
    x.append(get_representation_tweets(DIREC + LANG + ind))
    
df["content"] = pd.Series(x)
df.head()


# **User Behavior**
# - cross-post (#URL# tag)
# - retweet others (RT tag)
# - mentioning others (#USER# tag)
# - Amount of Hashtag

# In[9]:


# we cannot retreive the timestamp
# cross post
cp = "#URL#" 
cp_count = 0 
cross_post = []
for i in df["content"]:
    for j in i:
        cp_count += j.count(cp)
    cross_post.append(cp_count)
    cp_count = 0 
df["cross_post_duplicate"] = cross_post

# cross post version 2: including repeated cross post
cp = "#URL#" 
cp_count = 0 
cross_post = []
for i in df["content"]:
    for j in i:
        if cp in j:
            cp_count+=1
    cross_post.append(cp_count)
    cp_count = 0 

df["cross_post"] = cross_post


# In[10]:


# retweet others (RT tag)
rt_count = 0 
retweet_count = []
for i in df["content"]:
    for j in i:
        rt_count += j.count("RT")
    retweet_count.append(rt_count)
    rt_count = 0 
df["retweet"] = retweet_count


# In[11]:


# mentioning others (#USER# tag)
user_tag = 0
user_tag_counts = []
for i in df["content"]:
    for j in i:
        user_tag += j.count("#USER#")
    user_tag_counts.append(user_tag)
    user_tag = 0 
df["user_mention"] = user_tag_counts


# In[12]:


# hanshtag 
hashtag_count = 0
hashtag_counts = []
for i in df["content"]:
    for j in i:
        hashtag_count += j.count("#HASHTAG#")
    hashtag_counts.append(hashtag_count)
    hashtag_count = 0
df["hashtag"] = hashtag_counts
# df.head() 


# ### lexical feature
# - lexical diversity

# In[16]:


# lexical diversity
def clean(sentence):
    input_str = sentence
#     output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str) # remove punctiation
#     output_str = re.sub('\W+',' ',input_str) 
    output_str = re.sub('#URL#', '', input_str) # remove URL tag
    output_str = re.sub('RT', '', output_str) # remove RT tag
    output_str = re.sub('#USER#', '', output_str) # remove USER tag
    output_str = re.sub('#HASHTAG#', '', output_str) # remove HASHTAG tag
#     output_str = re.sub(' s ', ' ', output_str) # remove 's
#     output_str = re.sub('', output_str)

    return output_str


# In[17]:


nlp = spacy.load("en_core_web_sm")


# In[18]:


lemma = []
lexical_diversity = []
for text in df['content']:
    text = clean(''.join(text))
    doc = nlp(text)
    for token in doc:
        lemma.append(token.lemma_)
    lexical_diversity.append(ld.ttr(lemma))
    lemma = []


# In[19]:


df['lexical_diversity'] = lexical_diversity
# df.head()


# ### Semantic features
# - number of exclamation marks
# - number of question marks
# - number of named entities

# In[20]:


# number of exclamation marks
count = 0
exclamation_mark = []
for i in df['content']:
    for j in i:
        count += j.count('!')
    exclamation_mark.append(count)
    count = 0
df['exclamation_mark'] = exclamation_mark


# In[21]:


# number of question marks
q = 0
question_mark = []
for i in df['content']:
    for j in i:
        q += j.count('?')
    question_mark.append(q)
    q = 0
df['question_mark'] = question_mark


# In[23]:


# Named Entities
name_entities = []
for text in df['content']:
    text = clean(''.join(text))
    doc = nlp(text)
    name_entities.append(len(doc.ents))


# In[24]:


df['name_entites'] = name_entities
# df.head()


# ### syntactic features
# - adjective frequency
# - verb frequency
# - noun frequency
# - adverb frequency
# - pronoun frequency

# In[25]:


# adj frequency


# In[26]:


c = 0
adj_freq = []
for i in df['content']:
    for j in i:
        j = clean(j)
        doc = nlp(j)
        for token in doc:
            if token.pos_ == 'ADJ':
                c += 1
    adj_freq.append(c)
    c = 0


# In[27]:


df['adjective_frequecy'] = adj_freq
# df.head()


# In[28]:


# verb frequency
v = 0
verb_freq = []
for i in df['content']:
    for j in i:
        j = clean(j)
        doc = nlp(j)
        for token in doc:
            if token.pos_ == 'VERB':
                v += 1
    verb_freq.append(v)
    v = 0
# verb_freq


# In[29]:


df['verb_frequency'] = verb_freq


# In[30]:


# noun frequency
n = 0
noun_freq = []
for i in df['content']:
    for j in i:
        j = clean(j)
        doc = nlp(j)
        for token in doc:
            if token.pos_ == 'NOUN':
                n += 1
    noun_freq.append(n)
    n = 0
# noun_freq


# In[31]:


df['noun_freq'] = noun_freq


# In[32]:


# adv frequency
c = 0
adv_freq = []
for i in df['content']:
    for j in i:
        j = clean(j)
        doc = nlp(j)
        for token in doc:
            if token.pos_ == 'ADV':
                c += 1
    adv_freq.append(c)
    c = 0
# adv_freq


# In[33]:


df['adv_freq'] = adv_freq


# In[34]:


# pronoun frequency
c = 0
pron_freq = []
for i in df['content']:
    for j in i:
        j = clean(j)
        doc = nlp(j)
        for token in doc:
            if token.pos_ == 'PRON':
                c += 1
    pron_freq.append(c)
    c = 0
df['pronoun_frequency'] = pron_freq


# In[38]:


df = df.rename(columns={'noun_freq':'noun_frequency', 'adv_freq':'adverb_frequency', 'name_entites': 'named_entities'})
# df


# ### sentiment analysis
# - mean of sentiment values
# - standard deviation of sentiment values
# - mean of subjectivity scores
# - standard deviation of subjectivity scores

# In[57]:


sentiment_per_tweet = []
subjectivity_per_tweet = []
mean_sentiment_per_user = []
std_sentiment_per_user = []
mean_subjectivity_per_user = []
std_subjectivity_per_user = []
for i in df['content']:
    for j in i:
        j = clean(j)
        sentiment_per_tweet.append(TextBlob(j).sentiment.polarity)
        subjectivity_per_tweet.append(TextBlob(j).sentiment.subjectivity)
    mean_sentiment_per_user.append(np.mean(sentiment_per_tweet)) # the average sentiment of each user
    std_sentiment_per_user.append(np.std(sentiment_per_tweet))# the standard deviation of sentiment of each user
    mean_subjectivity_per_user.append(np.mean(subjectivity_per_tweet)) # the average subjectivity value of each user
    std_subjectivity_per_user.append(np.std(subjectivity_per_tweet)) # the standard deviation of subjectivity of each user
    sentiment_per_tweet = []
    subjectivity_per_tweet = []


# In[64]:


df['mean_sentiment'] = mean_sentiment_per_user
df['std_of_sentiment'] = std_sentiment_per_user
df['mean_subjectivity'] = mean_subjectivity_per_user
df['std_of_subjectivity'] = std_subjectivity_per_user


# In[65]:


# df.head()


# In[66]:


df.to_csv("dataframe_with_sentiment.csv", index = False)


# In[71]:


# fakedf['ID'][df['label']==0].count()


# In[ ]:




