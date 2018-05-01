
#Document Classification : positive vs negative

import nltk
from nltk.corpus import movie_reviews
from nltk import word_tokenize
import random

docs = [(movie_reviews.words(fileid), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

random.shuffle(docs)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_set = [ word for word in all_words ]
word_features = word_set[:2000]  #2000개까지만 feature 로 쓰기로 설정

def doc_features(document):
    doc_words = set(document)
    features = {}
    for word in word_features:
        features[word]= (word in doc_words) #This should appear as Boolean value
    return features  
#returns dictionary

doc_features(docs[0][0])   #docs[0][0] ==> movie_reviews의 첫번째 카테고리의 첫번째 파일아이디 문서

featuresets = [ (doc_features(d),c) for (d,c) in docs]
train, test = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train)

print(nltk.classify.accuracy(classifier, test))

classifier.show_most_informative_features(5)
