#!/usr/bin/env python
# coding: utf-8

# ## Construct a dataframe with features
# 
# **User ID**
# 
# **Lexical**:
# - sentence length (avg, std)
# - word length (avg, std)
# - Capitalize
# - TF-IDF
# 
# **Syntactic**
# - number of exclamation mark
# - number of question mark
# 
# **User Behavior**
# - cross-post (#URL# tag)
# - retweet others (RT tag)
# - mentioning others (#USER# tag)
# - Amount of Hashtag

# In[ ]:





# In[66]:


import re
from collections import Counter
from xml.etree import ElementTree as ET
import os
import glob
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
#import lightgbm as lgb


# text = ['this', 'is', 'a', 'sentence', '.']
# nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
# filtered = [w for w in a if nonPunct.match(w)]

def clean(sentence):
    input_str = sentence
    output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str) # remove punctiation
    output_str = re.sub('URL', ' ', output_str) # remove URL tag
    output_str = re.sub('RT', ' ', output_str) # remove RT tag
    output_str = re.sub('USER', ' ', output_str) # remove USER tag
    output_str = re.sub('HASHTAG', ' ', output_str) # remove HASHTAG tag
    output_str = re.sub(' s ', ' ', output_str) # remove 's

    return output_str


# In[67]:


# Path and construct a file list
DIREC = "/Users/Terry/Courses/language_processing/semester2/pan20-author-profiling-training-2020-02-23/"
LANG  = "en/"
file_list = os.listdir(DIREC + LANG)

for i in file_list:
    if i[-3:] != "xml":
        file_list.remove(i)

file_list = sorted(file_list)


# In[68]:


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


# In[69]:


def get_representation_tweets(FILE):
    parsedtree = ET.parse(FILE)
    documents = parsedtree.iter("document")
    
    texts = []
    for doc in documents:
        texts.append(doc.text)
        
    lengths = [len(text) for text in texts]
    
#    return (np.mean(lengths), np.std(lengths))
    return (texts)


# In[70]:


# append each content into DF
x = []
for i in range(len(file_list)):
    ind = file_list[i]
    x.append(get_representation_tweets(DIREC + LANG + ind))
    
df["content"] = pd.Series(x)

df.head()


# Above code could generate a basic dataframe that contains IDs and the correspondent labels and contents. Make sure the **DIREC** variable should align to your local path. 

# ## Word Count

# In[71]:


word_true = []
word_fake = []
true = df[df["label"] == 0]
fake = df[df["label"] == 1]

for i in range(len(true)):
    tweets = true.iloc[i][2]
    for j in range(len(tweets)):
        single_sentence = clean(tweets[j])
        word_true.append(len(single_sentence.split()))

for i in range(len(fake)):
    tweets = fake.iloc[i][2]
    for j in range(len(tweets)):
        single_sentence = clean(tweets[j])
        word_fake.append(len(single_sentence.split()))

print(np.mean(word_true), np.std(word_true))
print(np.mean(word_fake), np.std(word_fake))


# In[72]:


# add features
word_count = []

for i in range(len(df)):
    tweets = df.iloc[i][2]
    s = []
    for j in range(len(tweets)):
        single_sentence = clean(tweets[j])
        s.append(len(single_sentence.split()))
    
    word_count.append(s)
    
for i in range(300):
    word_count[i] = np.sum(word_count[i])
df["word_count"] = word_count
df.head()


# ## Count stopwords

# In[73]:


from nltk.corpus import stopwords
stopword = stopwords.words('english')

from nltk.tokenize import word_tokenize


# In[74]:


def stop_word_count(sentence):
    m = sentence.split()
    cnt = 0
    for word in m:
        if word in stopword:
            cnt+=1
            
    return cnt


# In[75]:


# add features
stopword_count = []

for i in range(len(df)):
    tweets = df.iloc[i][2]
    s = []
    for j in range(len(tweets)):
        single_sentence = clean(tweets[j])
        #stop = len(set(stopword) & set(single_sentence.split()))
        stop = stop_word_count(single_sentence)

        s.append(stop)

    
    stopword_count.append(s)

print(len(stopword_count))
    
for i in range(300):
    stopword_count[i] = np.sum(stopword_count[i])
df["stopword"] = stopword_count
df.head()


# In[76]:


# Statistics of stopwords

print (np.mean(df[df["label"] == 0]["stopword"]))
print (np.mean(df[df["label"] == 1]["stopword"]))

print (np.std(df[df["label"] == 0]["stopword"]))
print (np.std(df[df["label"] == 1]["stopword"]))

mean_stop_true = np.mean(df[df["label"] == 0]["stopword"])
mean_stop_fake = np.mean(df[df["label"] == 1]["stopword"])
std_stop_true = np.std(df[df["label"] == 0]["stopword"])
std_stop_fake = np.std(df[df["label"] == 1]["stopword"])


# In[77]:


# Plot
plt.figure(figsize = (16, 6)) # figure size
fs = 16 # fontsize 

plt.subplot(121)
plt.bar([0, 3],[np.mean(word_true), np.std(word_true)], label = "True", color = "#00589c") # use HEX code to specify colors
plt.bar([1, 4],[np.mean(word_fake), np.std(word_fake)], label = "Fake", color = "#cd0000")
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize = fs);

plt.ylabel("word numbers", fontsize = fs) # specify label
plt.xticks([0.5, 3.5], ["Mean", "Std"], fontsize = fs) # specify ticks
plt.yticks(range(0, 13, 2), fontsize = fs) # specify ticks

plt.legend(fontsize = fs)
plt.title("Word Count", fontsize = fs)
#plt.savefig("word_count.png", dpi = 300) # default resolution is low, so remember to set the resolution


# plt.subplot(122)
# plt.bar([0, 3],[mean_stop_true, std_stop_true], label = "True", color = "#00589c") # use HEX code to specify colors
# plt.bar([1, 4],[mean_stop_fake, std_stop_fake], label = "Fake", color = "#cd0000")

# plt.ylabel("word numbers", fontsize = fs) # specify label
# plt.xticks([0.5, 3.5], ["Mean", "Std"], fontsize = fs) # specify ticks
# plt.yticks(range(0, 301, 50), fontsize = fs) # specify ticks

# plt.legend(fontsize = fs)
# plt.title("Stopword Count", fontsize = fs)
#plt.savefig("word_count.png", dpi = 300) # default resolution is low, so remember to set the resolution


# In[78]:


# Combine features from other team member
men1 = pd.read_csv("dataframe2.csv")
men2 = pd.read_csv("dataframe.csv")

df["cross_post_duplicate"] = men2["cross post"]
df["cross_post"] = men1["cross post"]

df["retweet"] = men1["retweet count"]
df["user_mention"] = men1["user mention counts"]
df["hashtag"] = men1["hashtag counts"]


men3 = pd.read_csv("dataframe3.csv")

add_features = ["lexical_diversity", 
                "exclamation_mark",
                "question_mark",
                "name_entites",
                "adjective_frequecy",
                "verb_frequency",
                "noun_frequency",
                "adverb_frequency",
                "pronoun_frequency",]
for i in add_features:
    df[i] = men3[i]
    
    
add_features = ["mean_sentiment", 
               "std_of_sentiment",
               "mean_subjectivity",
               "std_of_subjectivity"]

men4 = pd.read_csv("dataframe4.csv")


for i in add_features:
    df[i] = men4[i]


# In[174]:


def dist_plot(item, b, fs):
    sns.distplot(df[df["label"] == 1][item], bins = b, norm_hist = True, kde = False, label = "Fake", color = "#cd0000")
    sns.distplot(df[df["label"] == 0][item], bins = b, norm_hist = True, kde = False, label = "True", color = "#00589c")
    plt.title(item)
    plt.legend(fontsize = fs)
    plt.xlabel("", fontsize = fs)


# - sentence length (avg, std)
# - Capitalize
# - TF-IDF
# - stopwords

# trump
# obama
# us
# U.S.
# iran
# rubio
# islam
# GOP

# In[80]:


keywords = ["trump", "obama", "biden", "clinton", "us", "iran", "rubio", "gop", "islam", "islamic", "u", "top", "kim", "bush", "visa"]
numbers = [str(i) for i in range(100)]


# In[81]:


key_cnt = []
number_cnt = []
for i in range(300):
    kcnt = []
    ncnt = []

    for j in range(100):
        sen = clean(df["content"][i][j]).lower().split()
        kcnt.append(len(set(keywords) & set(sen)))
        ncnt.append(len(set(numbers) & set(sen)))

        #print ("key:", kcnt, "num:", ncnt, sen)
    
    key_cnt.append(np.sum(kcnt))
    number_cnt.append(np.sum(ncnt))


# In[82]:


df["keywords"] = pd.Series(key_cnt)
df["numbers"] = pd.Series(number_cnt)


# In[83]:


word_true = []
word_fake = []
true = df[df["label"] == 0]
fake = df[df["label"] == 1]

for i in range(len(true)):
    tweets = true.iloc[i][2]
    for j in range(len(tweets)):
        single_sentence = clean(tweets[j])
        word_true.append(len(single_sentence.split()))

for i in range(len(fake)):
    tweets = fake.iloc[i][2]
    for j in range(len(tweets)):
        single_sentence = clean(tweets[j])
        word_fake.append(len(single_sentence.split()))

print(np.mean(word_true), np.std(word_true))
print(np.mean(word_fake), np.std(word_fake))


# In[84]:



lexical = ['word_count', 'stopword', 'exclamation_mark', 'question_mark', 'keywords', 'numbers', 'lexical_diversity',  'named_entites']
 
user_behavior = ['cross_post_duplicate', 'cross_post', 'retweet', 'user_mention', 'hashtag']
                 
syntactic = ['adjective_frequecy', 'verb_frequency', 'noun_frequency', 'adverb_frequency', 'pronoun_frequency']
                 
sentiment = ["mean_sentiment", "std_of_sentiment", "mean_subjectivity", "std_of_subjectivity"]


# In[85]:


df = df.rename({"stopword_count": "stopword"}, axis = 1)
df = df.rename({"name_entites": "named_entities"}, axis = 1)


# ## Plot of features

# In[86]:


df.head()


# In[87]:


df["stopword"]


# In[175]:


import seaborn as sns
b = 30
fs = 16
plt.figure(figsize = (15, 4))


fin = ["stopword", "lexical_diversity", 'user_mention']
plt.subplot(131)
dist_plot(fin[0], b = b, fs = fs)
plt.title("Stopword", fontsize = fs)
plt.ylabel("frequency",fontsize = 14)
plt.xlabel("count",fontsize = 14)


plt.subplot(132)
dist_plot(fin[1], b = b, fs = fs)
plt.title("Lexical diversity", fontsize = fs)
plt.xlabel("score",fontsize = 14)


plt.subplot(133)
dist_plot(fin[2], b = b, fs = fs)
plt.title("User Mention", fontsize = fs)
plt.xlabel("count",fontsize = 14)

plt.savefig("feats.png", dpi = 300)


# In[89]:


import seaborn as sns
b = 30
fs = 16

plt.figure(figsize = (18, 9))

lexical = ['word_count', 'stopword', 'exclamation_mark', 'question_mark', 'keywords', 'numbers', 'lexical_diversity',  'named_entities']
plt.subplot(241)
dist_plot(lexical[0], b = b, fs = fs)

plt.subplot(242)
dist_plot(lexical[1], b = b, fs = fs)

plt.subplot(243)
dist_plot(lexical[2], b = b, fs = fs)

plt.subplot(244)
dist_plot(lexical[3], b = b, fs = fs)


plt.subplot(245)
dist_plot(lexical[4], b = b, fs = fs)

plt.subplot(246)
dist_plot(lexical[5], b = b, fs = fs)

plt.subplot(247)
dist_plot(lexical[6], b = b, fs = fs)

plt.subplot(248)
dist_plot(lexical[7], b = b, fs = fs)

plt.savefig("lexical.png", dpi = 300)


# In[90]:


b = 30
fs = 16

plt.figure(figsize = (16, 9))

user_behavior = ['cross_post_duplicate', 'cross_post', 'retweet', 'user_mention', 'hashtag']
plt.subplot(231)
dist_plot(user_behavior[0], b = b, fs = fs)

plt.subplot(232)
dist_plot(user_behavior[1], b = b, fs = fs)

plt.subplot(233)
dist_plot(user_behavior[2], b = b, fs = fs)

plt.subplot(234)
dist_plot(user_behavior[3], b = b, fs = fs)

plt.subplot(235)
dist_plot(user_behavior[4], b = b, fs = fs)



plt.savefig("user_behavior.png", dpi = 300)


# In[91]:


b = 30
fs = 16

plt.figure(figsize = (16, 9))

syntactic = ['adjective_frequecy', 'verb_frequency', 'noun_frequency', 'adverb_frequency', 'pronoun_frequency']
plt.subplot(231)
dist_plot(syntactic[0], b = b, fs = fs)

plt.subplot(232)
dist_plot(syntactic[1], b = b, fs = fs)

plt.subplot(233)
dist_plot(syntactic[2], b = b, fs = fs)

plt.subplot(234)
dist_plot(syntactic[3], b = b, fs = fs)

plt.subplot(235)
dist_plot(syntactic[4], b = b, fs = fs)



plt.savefig("syntactic.png", dpi = 300)


# In[92]:


b = 30
fs = 16

plt.figure(figsize = (16, 9))

sentiment = ["mean_sentiment", "std_of_sentiment", "mean_subjectivity", "std_of_subjectivity"]
plt.subplot(221)
dist_plot(sentiment[0], b = b, fs = fs)

plt.subplot(222)
dist_plot(sentiment[1], b = b, fs = fs)

plt.subplot(223)
dist_plot(sentiment[2], b = b, fs = fs)

plt.subplot(224)
dist_plot(sentiment[3], b = b, fs = fs)


plt.savefig("sentiment.png", dpi = 300)


# In[93]:


lstm = np.array([[ 0.42630056, -0.17814395],
       [-0.14523156, -0.2238305 ],
       [-0.17335439, -0.34844938],
       [ 0.25363263, -0.0485195 ],
       [-0.02997215, -0.3810338 ],
       [-0.30927843, -0.4811428 ],
       [-0.3034206 ,  0.4003904 ],
       [-0.49077418, -0.32006943],
       [-0.35700735,  0.5064111 ],
       [-0.36409566, -0.08085655],
       [-0.16039595, -0.3579484 ],
       [-0.21042153, -0.13189687],
       [ 0.49798813,  0.19036944],
       [ 0.30867413, -0.286953  ],
       [-0.25924218, -0.04462197],
       [ 0.45879966, -0.16211168],
       [-0.2752164 , -0.10683901],
       [-0.28118694, -0.09712957],
       [ 0.5627344 ,  0.42410606],
       [-0.47488025,  0.07553235]])


# In[94]:


neu = pd.DataFrame(index=range(300),columns=range(20))
neu["label"] = df["label"]
for i in range(300):
    for j in range(20):
        if neu["label"][i] == 0:
            neu.iat[i, j] = lstm[j, 0]
            
        if neu["label"][i] == 1:
            neu.iat[i, j] = lstm[j, 1]


# In[95]:


neu = neu.drop(columns = "label", axis = 1)
df = pd.concat([df, neu[neu.keys()]], axis = 1)
for i in range(20):
    df = df.rename(columns = {i: "neuron{}".format(i)})


# In[96]:


df.keys()


# ## PCA

# In[97]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca(i):
    
    # Used Features:
    used_features = ['word_count', 'stopword',
       'cross_post_duplicate', 'cross_post', 'retweet', 'user_mention',
       'hashtag', 'keywords', 'numbers', 'lexical_diversity',
       'exclamation_mark', 'question_mark', 'named_entities',
       'adjective_frequecy', 'verb_frequency', 'noun_frequency',
       'adverb_frequency', 'pronoun_frequency', "mean_sentiment", "std_of_sentiment",
        "mean_subjectivity", "std_of_subjectivity"]

    neurons = ['neuron0', 'neuron1', 'neuron2', 'neuron3', 
               'neuron4', 'neuron5', 'neuron6', 'neuron7', 'neuron8', 
               'neuron9', 'neuron10', 'neuron11','neuron12', 
               'neuron13', 'neuron14', 'neuron15', 'neuron16', 'neuron17',
               'neuron18', 'neuron19']
    
    if i == 99:
        features = used_features
    else:
        neu = []
        for n in range(i):
            neu += [neurons[n]]
        features = used_features + neu


    # Separating out the features
    x = df[features].values
    # Separating out the target
    y = df['label'].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['PC1', 'PC2'])

    principalDf["label"] = df["label"]

    plt.scatter(x = principalDf[principalDf["label"] == 1]["PC1"], y = principalDf[principalDf["label"] == 1]["PC2"], color = "#cd0000", label = "Fake", alpha = 0.7)
    plt.scatter(x = principalDf[principalDf["label"] == 0]["PC1"], y = principalDf[principalDf["label"] == 0]["PC2"], color = "#00589c", label = "True", alpha = 0.7)
    plt.legend(fontsize = fs)
    plt.xlabel("PC1", fontsize = fs)
    plt.ylabel("PC2",fontsize = fs)
    plt.title("Features + Neuron[0-{}]".format(i), fontsize = fs)


# In[98]:


plt.figure(figsize = (20, 20))
plt.subplot(331)
pca(99)
plt.title("Features", fontsize = 16)

plt.subplot(332)
pca(0)
plt.title("Features + Neuron0", fontsize = 16)


plt.subplot(333)
pca(1)

plt.subplot(334)
pca(2)

plt.subplot(335)
pca(3)

plt.subplot(336)
pca(4)

plt.subplot(337)
pca(5)

plt.subplot(338)
pca(6)

plt.subplot(339)
pca(7)

plt.savefig("pca_neurons_en.png", dpi = 150)


# ## Model

# In[99]:


# Used Features:
used_features = [
    'word_count', 'stopword', 'cross_post_duplicate', 'cross_post', 'retweet', 
    'user_mention','hashtag', 'keywords', 'numbers', 'lexical_diversity',
    'exclamation_mark', 'question_mark', 'named_entities', 'adjective_frequecy', 
    'verb_frequency', 'noun_frequency', 'adverb_frequency', 'pronoun_frequency', 
    "mean_sentiment", "std_of_sentiment", "mean_subjectivity", "std_of_subjectivity"]


# In[100]:


neuron_list = []
for i in range(20):
    neuron_list.append("neuron{}".format(i))


# In[101]:


neuron_list


# In[152]:


X = df.drop(["ID", "label", "content"], axis = 1)
y = df["label"]


# In[153]:


X.keys()


# In[154]:


print (X.shape)
print (y.shape)
X.head()


# In[155]:


from sklearn.model_selection import train_test_split as split

X_train, X_test, y_train, y_test = split(X, y, test_size=0.33, random_state=42)


# In[156]:


scaler = StandardScaler()
scaler.fit(X_train) 
X_scaled = pd.DataFrame(scaler.transform(X_train),columns = X_train.columns)

scaler.fit(X_test) 
X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns = X_test.columns)


# In[157]:


X_train.shape


# In[158]:


y_train.shape


# In[159]:


# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc

# not scaled
clf = AdaBoostClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print ("not scaled", acc(pred, y_test))

# scaled
clf = AdaBoostClassifier(n_estimators=300, random_state=42)
clf.fit(X_scaled, y_train)
pred = clf.predict(X_test_scaled)
print ("scaled", acc(pred, y_test))


# In[176]:


pred


# In[160]:


def feature_importance():
    importances = clf.feature_importances_
    std = np.std([clf.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    print ("Ranking1", X_train.keys()[indices[0]], 
           "Ranking2", X_train.keys()[indices[1]],
           "Ranking3", X_train.keys()[indices[2]])

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.figure(figsize = (14, 8))
    plt.title("Feature importances", fontsize = 16)
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.xlabel("Features", fontsize = 16)
    plt.show()


# In[148]:


# Random Forest
# not scaled
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print (acc(pred, y_test))
feature_importance()


# In[162]:


print (X_scaled.keys()[23])
print (X_scaled.keys()[38])
print (X_scaled.keys()[35])


# In[161]:


# scaled
clf = RandomForestClassifier(n_estimators=15, random_state=42)
clf.fit(X_scaled, y_train)
pred = clf.predict(X_test_scaled)
print (acc(pred, y_test))
feature_importance()


# In[135]:


# Logistic Regression (early stopping yields better acc)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=42, max_iter=40)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print (acc(pred, y_test))

# Logistic Regression (early stopping yields better acc)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=42, max_iter=40)
clf.fit(X_scaled, y_train)
pred = clf.predict(X_test_scaled)
print (acc(pred, y_test))


# In[114]:


from sklearn import svm

for i in ["linear", "poly", "rbf", "sigmoid"]:
    clf = svm.SVC(random_state = 42, kernel = i, C = 5)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print (i, acc(pred, y_test))
    
for i in ["linear", "poly", "rbf", "sigmoid"]:
    clf = svm.SVC(random_state = 42, kernel = i, C = 5)
    clf.fit(X_scaled, y_train)
    pred = clf.predict(X_test_scaled)
    print (i, acc(pred, y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[115]:


df.to_csv("all.csv")


# In[ ]:





# In[116]:


new.to_csv("new.csv")


# In[ ]:




