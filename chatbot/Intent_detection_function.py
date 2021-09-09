import random
random.seed(42)
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib
import keras
import sys

from entity_function import getEntities


def Intent_detection_function(keyboard):

    mlp_greeting = keras.models.load_model('model_greetings.h5')
    mlp_search = keras.models.load_model('model_search.h5')
    mlp_suggestion = keras.models.load_model('model_suggestion.h5')
    mlp_farewell = keras.models.load_model('model_farewell.h5')
    cv = joblib.load("vectorizer.pkl")


    input_text = pd.DataFrame(data={'Sentence': [keyboard]})
    test_proc = cv.transform(input_text['Sentence'])
    #df_test_proc, test_proc = processing(test, cv=cv)

    #test_proc = test_proc.toarray()

    gret_prob = mlp_greeting.predict(test_proc)

    search_prob = mlp_search.predict(test_proc)
    sugg_prob = mlp_suggestion.predict(test_proc)
    fare_prob = mlp_farewell.predict(test_proc)


    probs = [gret_prob, search_prob, sugg_prob, fare_prob]
    print(probs)
    idx = np.argmax(probs)

    if idx == 0:
        intent = "Greeting"
        keyword = None
    elif idx == 1:
        intent = "Search"
        keyword = getEntities(keyboard)
    elif idx == 2:
        intent = "Suggestions"
        keyword = getEntities(keyboard)
    elif idx == 3:
        intent = "Farewell"
        keyword = None

    return intent, keyword
