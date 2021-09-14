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
import tensorflow.keras as keras
import sys
import scipy

from entity_function import getEntities


def Intent_detection_function(keyboard, df):
    keyword = getEntities(keyboard)

    def coincidence(x):
        if x['Sentence'] == keyboard:
            return x.name
        else:
            return 0

    if df['Sentence'].str.contains(keyboard, case=True).any():
        index = df.apply(lambda x: coincidence(x), axis=1)
        intent = df.loc[np.argmax(index.to_numpy()), 'Intent type']
        return intent, keyword

    if keyword != 0:
        search_plus = 0.3
        sugg_plus = 0.3
        gret_plus = 0
        fare_plus = 0
        keyboard = keyboard.replace(keyword, '')


        def remove_entity(x):
            if "entity" in x['Sentence']:
                x['Sentence'] = x['Sentence'].replace('entity', '')
            return x

        df_new = df.apply(lambda x: remove_entity(x), axis=1)
    else:
        df_new = df
        gret_plus = 0.3
        fare_plus = 0.3
        search_plus = 0
        sugg_plus = 0

    if df_new['Sentence'].str.contains(keyboard, case=True).any():
        index = df_new.apply(lambda x: coincidence(x), axis=1)
        intent = df.loc[np.argmax(index.to_numpy()), 'Intent type']
        if keyword != 0:
            return intent, keyword
        else:
            return intent, keyword

    else:

        mlp_greeting = keras.models.load_model('model_greetings.h5')
        mlp_search = keras.models.load_model('model_search.h5')
        mlp_suggestion = keras.models.load_model('model_suggestion.h5')
        mlp_farewell = keras.models.load_model('model_farewell.h5')
        cv = joblib.load("vectorizer.pkl")

        input_text = pd.DataFrame(data={'Sentence': [keyboard]})
        test_proc = cv.transform(input_text['Sentence'])
        # df_test_proc, test_proc = processing(test, cv=cv)

        # test_proc = test_proc.toarray()
        scipy.sparse.csr_matrix.sort_indices(test_proc)

        gret_prob = mlp_greeting.predict(test_proc)

        search_prob = mlp_search.predict(test_proc)
        sugg_prob = mlp_suggestion.predict(test_proc)
        fare_prob = mlp_farewell.predict(test_proc)

        probs = [gret_prob + gret_plus, search_prob + search_plus,
                 sugg_prob + sugg_plus, fare_prob + fare_plus]
        print(probs)
        idx = np.argmax(probs)

        if idx == 0:
            intent = "Greeting"
            keyword = None
        elif idx == 1:
            intent = "Search"
        elif idx == 2:
            intent = "Suggestions"
        elif idx == 3:
            intent = "Farewell"
            keyword = None

    return intent, keyword