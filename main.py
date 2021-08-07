import pickle
import tensorflow as tf
import numpy as np
import pickle
import json

from tensorflow.python.ops.gen_array_ops import reverse

from nlp_pipelines import nltk_POS_lemmatizer

model_path = 'adam_model'
model = tf.keras.models.load_model(model_path)

words = pickle.load(open(f'{model_path}/words.pkl', 'rb'))
classes = pickle.load(open(f'{model_path}/classes.pkl', 'rb'))

def main():
    done = False
    while not done:
        sentence = input("Talk: ")
        if sentence.lower() == "exit":
            done = True
        else:
            bag = bag_words(sentence, words)
            result = model.predict(np.array([bag]))[0]
            probs = {classes[i]: prob for i, prob in enumerate(result)}
            probs_top_three = {k: f"{v*100:,.2f}%" for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}
            print(probs_top_three)
            print(classes[np.argmax(result)])


def bag_words(sentence, known_words):
    bag = [0] * len(known_words)

    word_pattern = nltk_POS_lemmatizer(sentence)

    for new_word in word_pattern:
        for i, word in enumerate(known_words):
            if new_word == word:
                bag[i] = 1

    return np.array(bag)


if __name__ == "__main__":
    main()
