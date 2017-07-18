import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# data load
searches = pd.read_csv("search-terms-clean.csv")

# print searches.head(3)

# Topic Modeling with Non-Negative Matrix Factorization (NMF)
vectorizer = TfidfVectorizer(stop_words='english')
document_term_mat = vectorizer.fit_transform(searches['Search term'].values)

# model result:
feature_words = vectorizer.get_feature_names()

# checking
# feature_words[0:10] # first 10 words
# document_term_mat.todense()/document_term_mat.todense().min()

# Topic Model with 3 topics
n_topics = 3
nmf = NMF(n_components=n_topics)

W_sklearn = nmf.fit_transform(document_term_mat)
H_sklearn = nmf.components_

# functions for calculating the mean sqaured error:
def reconst_mse(target, left, right):
	'''
	INPUT: 
		target = document term matrix (X) which should be dot product of W and H
		left = W matrix
		right = H matrix 
	OUTPUT:
		mean squared error ( target - (left.dot(right)) ** 2).mean()
	'''
    
    return (array(target - left.dot(right))**2).mean()


def describe_nmf_results(document_term_mat, W, H, n_top_words = 15):
	'''
	INPUT:
		Document term matrix (X)
		left matrix (W)
		right matrix (H)
		no. of top_words to be displayed in topics (default: 15)
	OUTPUT:
		Feature words from each topic 
	'''
    print("Reconstruction error: %f") %(reconst_mse(document_term_mat, W, H))

    for topic_num, topic in enumerate(H):
        print("Topic %d:" % topic_num)
        print(" ".join([feature_words[i] \
                for i in topic.argsort()[:-n_top_words - 1:-1]]))
    return

# print the NMF results:
print describe_nmf_results(document_term_mat, W_sklearn, H_sklearn, n_top_words = 15)

# **  Adding the topic # to the dataframe **

# Function to classify topics for each searchterm
def classify_topic(W_values):
	'''
	INPUT: 
		W values for each row (i.e. each topic values)
	OUTPUT:
		Topic # based on the dominating topic

	'''
    if W_values[0] > W_values[1]:
        return int(0) 
    elif W_values[1] > W_values[2]:
        return int(1)
    else:
        return int(2)

# Topics for each searchterm
topic_classes = []
for i in xrange(len(W_sklearn)):
    topic_classes.append(classify_topic(W_sklearn[i]))

# Adding the column "Topic" to the dataframe
searches['Topic'] = [i for i in topic_classes]

# print searches.head(2)

# Let's calculate Clicks, Impressions by Topic
topic_clicks = searches.groupby(by='Topic').Clicks.sum().reset_index()
topic_impressions = searches.groupby(by='Topic').Impressions.sum().reset_index()
topic_cost = searches.groupby(by='Topic').Cost.sum().reset_index()
topic_conversions = searches.groupby(by='Topic').Conversions.sum().reset_index()
topic_avg_postion = searches.groupby(by='Topic')['Avg. position'].mean().reset_index()
topic_df = pd.merge(pd.merge(pd.merge(topic_clicks, topic_impressions, on='Topic'), pd.merge(topic_cost,  topic_conversions, on='Topic'), on = 'Topic'), topic_avg_postion, on = 'Topic')
topic_df.head(2)

# Calculate CTR, Cost per conversion, Conversion rate
topic_df['CTR'] = (topic_df['Clicks'] * 1.0 / topic_df['Impressions']) * 100
topic_df['Avg_CPC'] = topic_df['Cost'] * 1.0 / topic_df['Clicks']
topic_df['Cost_per_Conversion'] = topic_df['Cost'] * 1.0 / topic_df['Conversions']
topic_df['Conv_rate'] = (topic_df['Conversions'] * 1.0 / topic_df['Clicks']) * 100

# Create Bins
searches['bins'] = pd.qcut(searches['Avg. position'], 4, labels=False)

avg_bin_values = searches['Avg. position'].groupby(cuts).mean().to_dict()
searches['Transformed_avg_position'] = searches['bins'].apply(lambda x: avg_bin_values[x])

# Linear Regression Model
features = ['Topic', 'Transformed_avg_position']
X = searches[features]
y = searches['Conv. rate']

# Test-Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Error functions:
def score(y_test,y_predict):
    log_diff = np.log(y_predict+1) - np.log(y_test+1)
    return np.sqrt(np.mean(log_diff**2))

def rmse(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict) 
    return np.sqrt(1.0* sum(( y_test - y_predict) ** 2 ) / len(y_test))

# Train model
regr = LinearRegression()
regr.fit(X_train, y_train)

#Predict on test model
test_predicted = regr.predict(X_test)
print score(y_test, test_predicted)
print rmse(y_test, test_predicted)

# Cross-Validation
n = 10
kf = KFold(n_splits=n)
model = LinearRegression()
train_error = np.empty(n)
test_error = np.empty(n)
for i, (train, test) in enumerate(kf.split(X)):
	model.fit(X.iloc[train], y[train])
    train_error[i] = rmse(y[train], model.predict(X.iloc[train]))
    test_error[i] = rmse(y[test], model.predict(X.iloc[test]))

print "Train Error:", np.mean(train_error)
print "Test Error:", np.mean(test_error)




