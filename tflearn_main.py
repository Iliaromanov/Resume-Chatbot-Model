import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
import random
import json
import pickle

# Prepare a stemmer
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Load the data
with open('intents.json') as file:
    data = json.load(file)

# Don't parse/preprocess data if already done before
try:
    with open('data.pickle', 'rb') as f:
        words, classes, training, output = pickle.load(f)
except:
    words = [] # possible unique words
    classes = [] # possible unique classes (tags)
    docs_x = [] # Pattern inputs
    docs_y = [] # Corresponding tags for docs_x patterns

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            pattern_wrds = nltk.word_tokenize(pattern) # breaks up string into list of words
            words.extend(pattern_wrds) # extend() adds all elements of an iterable to the end of the list
            docs_x.append(pattern_wrds) # If a pattern > one word then stemmer will just lowercase it
            docs_y.append(intent["tag"])

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

    # Stem words and convert to lowercase
    words = [stemmer.stem(w) for w in words]
    # Use set to remove duplicates and then sort
    words = sorted(list(set(words)))
    classes = sorted(classes)

    # Using one-hot encoding to prepare data for model
    training = []
    output = []
    out_empty = [0 for _ in range(len(classes))]

    for i, doc in enumerate(docs_x):
        bag = []

        # Stem each word in pattern and remove common punctuation symbols
        wrds = [stemmer.stem(w) for w in doc if w not in "?!.,'"]

        for w in words:
            if w in doc:
                bag.append(1) # put 1 if word is in pattern
            else:
                bag.append(0) # 0 if not

        output_row = out_empty[:]
        output_row[classes.index(docs_y[i])] += 1

        training.append(bag)
        output.append(output_row)

    # Convert lists into numpy arrays so that they can be given as inputs to the model
    training = np.array(training)
    output = np.array(output)

# Save the training and test data
with open('data.pickle', 'wb') as f:
    pickle.dump((words, classes, training, output), f)

# Building Neural Net
# Input layer has len(x[0]) neurons (# of unique words)
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8, activation="relu") # hidden layer with 8 neurons
net = tflearn.fully_connected(net, 8, activation="relu") # hidden layer with 8 neurons
# Output layer has len(Y[0]) neurons (# of unique pattern tags/classes)
net = tflearn.fully_connected(net, len(output[0]), activation="sigmoid")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Don't train model if it already exists
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=780, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_words(sentence, known_words):
    bag = []

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    for new_w in sentence_words:
        for known_w in known_words:
            if new_w == known_w:
                bag.append(1)
            else:
                bag.append(0)

    return np.array(bag)


def chat():
    print("Start talking with the bot!\n (type quit to stop)")
    while True:
        user = input("You: ")
        if user.lower() == "quit":
            break
        
        bag = bag_words(user, words)
        print(bag)
        result = model.predict([bag])
        print(result)
        result_idx = np.argmax(result) # gives index of greates value in result
        tag = classes[result_idx]
        print(classes)
        print(tag)

chat()
