import random 

import joblib

import sys


#from utils import load_cinema_reviews

import random
random.seed(42)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve

import matplotlib.pyplot as plt

import random 

import joblib

import sys


#from utils import load_cinema_reviews

import random
random.seed(42)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve

import matplotlib.pyplot as plt



df = pd.read_excel(r"C:\Users\jimmy\Desktop\Copia_de_Seguridad\Keepcoding_2\TFB-SmartBot\chatbot\training_chatbot.xlsx")

finish = False
while finish == False:
    keyboard = input('\n')
    
    intent, keyword = Intent_model(keyboard)

    if intent == "Gretting":
        print("Hi what can I do for you?")

    elif intent =="Search":
        print("This is what I have found on {}".format(keyword))
    
    elif intent=="Suggestions":
        print("This is what I have found related to {}".format(keyword))


def chatbot():
    input_text = input()

    test = pd.DataFrame(data={'Sentence': [input_text]})
    df_test_proc, test_proc = processing(test, cv=cv)

    gret_prob = mlp_greeting.predict_proba(test_proc)
    search_prob = mlp_search.predict_proba(test_proc)
    sugg_prob = mlp_suggestion.predict_proba(test_proc)

    probs = [gret_prob, search_prob, sugg_prob]
    idx = np.argmax(probs)

    if idx == 0:
        print("Esto es un saludo")
    elif idx == 1:
        print("Esto es una búsqueda")
    else:
        print("Esto es una sugerencia")

    print('¿Hemos acertado?')

    respuesta = input()
    if (respuesta == 'No' or respuesta == 'no'):
        probs = np.delete(probs, idx)
        idx_2 = np.argmax(probs)

        if idx == 0:
            if idx_2 == 0:
                print("Esto es una búsqueda")
            else:
                print("Esto es una sugerencia")
        elif idx == 1:
            if idx_2 == 0:
                print("Esto es un saludo")
            else:
                print("Esto es una sugerencia")
        else:
            if idx_2 == 0:
                print("Esto es un saludo")
            else:
                print("Esto es una búsqueda")
    else:
        print('¡Genial! ¡Hemos acertado!')


