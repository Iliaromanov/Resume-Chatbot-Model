import tensorflow as tf

from flask import Flask, redirect, render_template, request, session
from flask_session import Session


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():

    model = tf.keras.models.load_model(r'model/adam_model_2')

    if request.method == "POST":
        pass
    else:
        pass
