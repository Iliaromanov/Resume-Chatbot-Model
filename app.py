import tensorflow as tf
from model.main import bag_words
# from model.nlp_pipelines import nltk_POS_lemmatizer
import pickle
import numpy as np

from flask import Flask, redirect, render_template, request, session
from flask_session import Session


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get_response", methods=["POST"])
def get_response():
    model_path = r'model/adam_model_2'
    model = tf.keras.models.load_model(model_path)
    words = pickle.load(open(f'{model_path}/words.pkl', 'rb'))
    classes = pickle.load(open(f'{model_path}/classes.pkl', 'rb'))

    sentence = request.args["msg"]
    bag = bag_words(sentence, words)

    result = model.predict(np.array([bag]))[0]
    probs = {classes[i]: prob for i, prob in enumerate(result)}
    probs_top_three = {k: f"{v * 100:,.2f}%" for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}

    print(probs)

    return {
        "all_predictions": {k: f"{v * 100:,.2f}%" for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)},
        "top_three_predictions": probs_top_three,
        "chatbot_response": None
    }


if __name__ == "__main__":
    app.run()
