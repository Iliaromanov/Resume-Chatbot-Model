import pickle
import tensorflow as tf
import numpy as np
import pickle
import json

from nlp_pipelines import nltk_POS_lemmatizer

model_path = './sgd_model'
model = tf.keras.models.load_model(model_path)

words = pickle.load(open('sgd_model/words.pkl', 'rb'))
classes = pickle.load(open('sgd_model/classes.pkl', 'rb'))

def bag_words(sentence, known_words):
    bag = [0] * len(known_words)

    word_pattern = nltk_POS_lemmatizer(sentence)

    for new_word in word_pattern:
        for i, word in enumerate(known_words):
            if new_word == word:
                bag[i] = 1

    return np.array(bag)

done = False
while not done:
    sentence = input("Talk: ")
    if sentence.lower() == "exit":
        done = True
    else:
        bag = bag_words(sentence, words)
        result = model.predict(np.array([bag]))[0]
        probabilities = {classes[i]: f"{prob*100:,.2f}%" for i, prob in enumerate(result)}
        print(probabilities)
        print(classes[np.argmax(result)])

