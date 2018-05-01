#Conditional FreqDist
cfd = nltk.ConditionalFreqDist( (genre, word) 
                              for genre in brown.categories()
                              for word in brown.words(categories=genre)
                              )
genres = ['news', 'religion', 'romance']
modals = ['will', 'pray', 'love', 'hate', 'must', 'kill']
cfd.tabulate(conditions=genres, samples=modals)

from nltk.corpus import inaugural
cfd_2 = nltk.ConditionalFreqDist((target, fileid[:4])
                                  for fileid in inaugural.fileids()
                                  for w in inaugural.words(fileid)
                                 for target in ['america', 'citizen']
                                 if w.lower().startswith(target)
                                )

cfd_2.plot()

from nltk.corpus import PlaintextCorpusReader as ptc
root = '/usr/share/dict'
wordlists = ptc(root, '.*')
wordlists.fileids()
wordlists.words()

from nltk.corpus import BracketParseCorpusReader as bpc
root = ''
file_pattern = r".*/wsj_.*\.mrg"

genre_word = [ (genre, word) 
             for genre in ['news', 'romance']
             for word in brown.words(categories=genre)]

cfd_3 = nltk.ConditionalFreqDist(genre_word)
cfd_3['romance']['love']
cfd_3['romance'].most_common(10)


#Wordlist Corpora
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    eng_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - eng_vocab
    return sorted(unusual)

len(unusual_words(inaugural.words(inaugural.fileids()[1])))

#Stopwords /words that are too common
from nltk.corpus import stopwords

print(stopwords.words('english'))

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content= [w for w in text if w.lower() not in stopwords]
    return len(content)/len(text)

content_fraction(nltk.corpus.reuters.words())

#The WordNet
from nltk.corpus import wordnet as wn
wn.synsets('motorcar')
wn.synset('car.n.01').lemma_names()
wn.synset('car.n.01').examples()
wn.synset('car.n.01').lemmas()   #==wn.lemmas('car')

motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
types_of_motorcar[0]

sorted(lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas())
