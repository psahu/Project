import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF

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

