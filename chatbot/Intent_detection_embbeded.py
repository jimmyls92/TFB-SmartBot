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

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

import pickle

from entity_function import getEntities


def Intent_detection_embbeded(keyboard):

    from_disk = pickle.load(open("tv_layer.pkl", "rb"))
    new_v = TextVectorization.from_config(from_disk['config'])
    new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_v.set_weights(from_disk['weights'])
    nlp_embbeded = keras.models.load_model('embbedings.h5')

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
        input_text = pd.DataFrame(data={'Sentence': [keyboard]})

        test_proc = new_v(input_text)
        probs = nlp_embbeded.predict(test_proc)
        probs = [probs[0] + gret_plus, probs[1] + search_plus,
                 probs[2] + sugg_plus, probs[3] + fare_plus,
                 probs[4], probs[5]]
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
