import pickle
import tensorflow as tf
import numpy as np
import pickle
import json

from nlp_pipelines import nltk_POS_lemmatizer

version = "0.0.1"



header = f"""
#################################################
#      Ilia's Resume Chat-bot; v {version}          #
#################################################
"""

def main():
    print(header)

    model_path = 'adam_model_2'
    model = tf.keras.models.load_model(model_path)

    words = pickle.load(open(f'{model_path}/words.pkl', 'rb'))
    classes = pickle.load(open(f'{model_path}/classes.pkl', 'rb'))

    model_path_2 = "svm_model"
    model_2 = pickle.load(open(rf'{model_path_2}\svm_model.pkl', 'rb'))
    words_2 = pickle.load(open(rf'{model_path_2}\svm_words.pkl', 'rb'))
    classes_2 = pickle.load(open(rf'{model_path_2}\svm_classes.pkl', 'rb'))

    keras_model_path = "keras_DNN_model"
    keras_model = tf.keras.models.load_model(keras_model_path)
    keras_words = pickle.load(open(rf'{keras_model_path}\words.pkl', 'rb'))
    keras_classes = pickle.load(open(rf'{keras_model_path}\classes.pkl', 'rb'))

    done = False
    while not done:
        sentence = input("Talk: ")
        if sentence.lower() == "exit":
            done = True
        else:
            bag = bag_words(sentence, words)
            result = model_2.predict(np.array([bag]))
            print(f"SVM Prediction: {result}")

            result = model.predict(np.array([bag]))[0]
            probs = {classes[i]: prob for i, prob in enumerate(result)}
            probs_top_three = {k: f"{v*100:,.2f}%" for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}
            print(f"DNN Prediction: {probs_top_three} => {classes[np.argmax(result)]}")

            # newest trained keras model. The naming is pretty bad cus adam_model_2 is also a keras model, its just an older version
            keras_bag = bag_words(sentence, keras_words)
            result = keras_model.predict(np.array([keras_bag]))[0]
            probs = {keras_classes[i]: prob for i, prob in enumerate(result)}
            probs_top_three = {k: f"{v*100:,.2f}%" for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}
            print(f"Newest DNN Prediction (this one is used in the web app): {probs_top_three} => {keras_classes[np.argmax(result)]}")


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



# "Whats that AI project you did" classifies as AI-Based-Desktop-Controller by DNN and greeting by SVM
# "How can you help me" classifies as Goodbye 73%, about chat-bot 10% by DNN and about chat-bot by SVM, newest DNN gets it right
# "I have to get going" only classfies as goodbye in newest DNN, svm and adam classify it as Golang-Blockchain project