import pandas as pd
import numpy as np
import math
import operator

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import word_tokenize

# data load
searches = pd.read_csv("search-terms-clean.csv")

campaigns = searches.Campaign.unique()

# Each campaign is a document. Let's find Tokenized documents:
# no of documents in the corpus = N = len(campaigns)
N = len(campaigns)
list_of_document_words = []  # It's tokenized documents containing the entire corpus in a list of list. Each list represent a campaign
tokenized_document = [] 
for document in campaigns:
    document = list(searches['Search keyword'][searches['Campaign'] == document])
    for sentence in document:
        for word in sentence.split(' '):
            tokenized_document.append(word)
    list_of_document_words.append(tokenized_document)

print len(list_of_document_words) # must be 68 = len(campaigns)

# TF-IDF
tfidf = TfidfVectorizer()
tfidfed = tfidf.fit_transform(list_of_document_words[0])

# Term Frequeny function of each word for the document
def term_frequency(word, tokenized_document):
	'''
	INPUT: 
		word, and a the tokenized document in this case, its the list of keywords representing one campaign

	OUTPUT:
		frequency of occurance of that word in the document(/campaign)

	'''
    return tokenized_document.count(term)

# Sublinear term frequency
def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)


# IDF function
def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

# TF-IDF for all documents:
def tfidf(tokenized_documents):
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

   
# Find term frequency for each campaign
tf = {}
for document in list_of_document_words:
	for word in document:
		tf[word] = term_frequency(word, document)

# TF-IDF
tfidf(list_of_document_words)

#
sorted_tf = sorted(tf.items(), key=operator.itemgetter(1))











