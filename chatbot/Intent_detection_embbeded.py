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
import scipy

from entity_function import getEntities


def Intent_detection_embbeded(keyboard):

    # mlp_greeting = keras.models.load_model('model_greetings.h5')
    # mlp_search = keras.models.load_model('model_search.h5')
    # mlp_suggestion = keras.models.load_model('model_suggestion.h5')
    # mlp_farewell = keras.models.load_model('model_farewell.h5')
    nlp_embbeded = keras.models.load_model('embbedings.h5')
    cv = joblib.load("vectorizer.pkl")


    input_text = pd.DataFrame(data={'Sentence': [keyboard]})
    test_proc = cv.transform(input_text['Sentence'])
    #df_test_proc, test_proc = processing(test, cv=cv)

    #test_proc = test_proc.toarray()
    scipy.sparse.csr_matrix.sort_indices(test_proc)

    probs = nlp_embbeded.predict(test_proc)

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
    elif idx == 4:
        intent = "Options"
        keyword = None
    elif idx == 5:
        intent = 'Headers'
        keyword = None

    return intent, keyword
