# Profiling fake news spreaders from twitter
Nowadays, huge amounts of news posts on social media are making it increasingly important to identify fake news from real news, as billions of people experience news and events from social media platforms like twitter. We have used various supervised learning methods like AdaBoost, Logistic Regression, Random forest, and Support Vector Machines (SVM) on the data to differentiate fake news spreaders from non fake news spreaders. For our machine learning models, we have artificially generated 22 features based on our linguistic and common sense knowledge, and we also have used a long short-term memory-based (LSTM) neural network to extract more features. With only artificially generated features, we achieved an accuracy of 0.66 using linear SVM, and a 0.63 accuracy using Random Forest classifier. However, after including the features extracted by LSTM-based network, we achieved an accuracy of 1.0 in all of our machine learning models. 

## Python modules used:
- regex 2.2.1
- numpy  1.18.1
- pandas 1.0.0
- nltk 3.4.4
- nltk stopwords dictionary: english
- keras 2.3.1
- spaCy 2.2.4
- textblob 0.15.3
- lexical-diversity 0.1.1

## Python version:
Python 3.7.6

## LSTM:
Running on Google colab
