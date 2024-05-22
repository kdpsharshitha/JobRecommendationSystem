import math
from collections import Counter
from utils.IDF import tokenize,calculate_idf 


def calculate_tfidf(document, unique_words, idf_values):
    tfidf_vector = {}

    for term in unique_words:
      tf = tokenize(document).count(term)
      tfidf_vector[term] = tf * idf_values[term]

    return tfidf_vector



