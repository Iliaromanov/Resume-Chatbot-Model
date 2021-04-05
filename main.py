import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json


with open('intents.json') as file:
    data = json.load(file)

words = [] # possible unique words
labels = [] # possible unique labels (tags)
docs_x = [] # Pattern inputs
docs_y = [] # Corresponding tags for docs_x patterns

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern) # breaks up string into list of words
        words.extend(wrds) # extend() adds all elements of an iterable to the end of the list
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stem words and convert to lowercase
words = [stemmer.stem(w.lower()) for w in words]
# Use set to remove duplicates and sort
words = sorted(list(set(words)))
labels = sorted(labels)