
import nltk
import pickle
import pandas as pd
import random

from nltk.corpus import movie_reviews

first_txt = movie_reviews.words(movie_reviews.fileids()[0])
first = nltk.Text(movie_reviews.words(movie_reviews.fileids()[0]))

for id in movie_reviews.fileids():
    num_chars = len(movie_reviews.raw(id))
    num_words = len(movie_reviews.words(id))
    num_sents = len(movie_reviews.sents(id))
    num_vocab = len(set(w.lower for w in movie_reviews.words(id)))
    
    print(round(num_chars/num_words), round(num_words/num_sents), id)


from nltk.corpus import brown
print(brown.categories())

fic_text = brown.words(categories='fiction')
fdist = nltk.FreqDist(w.lower() for w in fic_text)

modals = ['dear', 'if', 'love', 'revenge', 'victory']
for m in modals:
    print(m + ':', fdist[m], end=' ')
