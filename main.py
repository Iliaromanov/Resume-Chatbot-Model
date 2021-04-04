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

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern) # breaks up string into list of words
        words.extend(wrds) # extend() adds all elements of an iterable to the end of the list
        docs.append(pattern)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])
