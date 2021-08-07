"""
This file contains multiple data pipelines for preprocessing intent sentences.
The effectiveness of these pipelines is evaluated in main.ipynb
and the best is chosen to be used for preprocessing intents data for the chatbot DNN model
"""
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

from typing import List

# Necessary downloads
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


def nltk_stemmer(sentence: str) -> List[str]:
    stemmer = LancasterStemmer()
    tokens = nltk.word_tokenize(sentence)

    return [stemmer.stem(w) for w in tokens if w not in "?!,."]

def nltk_no_POS_lemmatizer(sentence: str) -> List[str]:
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(sentence)

    return [lemmatizer.lemmatize(w.lower()) for w in tokens if w not in "?!,."]

def nltk_POS_lemmatizer(sentence: str) -> List[str]:
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(sentence)
    token_tag_pairs = nltk.pos_tag(tokens)

    return [lemmatizer.lemmatize(token[0], tag_dict.get(token[1][0], wordnet.NOUN)).lower() 
            for token in token_tag_pairs if token[0] not in "?!,."]