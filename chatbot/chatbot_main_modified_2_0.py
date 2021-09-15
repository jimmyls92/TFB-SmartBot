import random

import joblib

import sys

import random
random.seed(42)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sklearn
import requests
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve


import warnings
warnings.filterwarnings("ignore")

from  Intent_detection_3_0 import Intent_detection
import wikipedia as wiki

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
finish = False
activated = False
first = True
intent = "Hello"

def preprocessing(df,stop_words=[]):
    def remove_punctuation(x):
        #x = " ".join(re.findall('[\w]+', x))
        import string
        for i in range(len(string.punctuation)):
            x = x.replace(string.punctuation[i], "")
        return x
    def lower_words(x):
        x = x.lower()
        return x

    df = df.apply(lambda x : lower_words(x))
    df = df.apply(lambda x : remove_punctuation(x))
    #df = df.apply(lambda x : remove_stopWords(x, stop_words))

    return df

df = pd.read_excel('database_intents.xlsx', engine='openpyxl')
df['Sentence'] = preprocessing(df['Sentence'])


def generate_text(keyboard):
    global finish
    global activated
    global first
    global intent
    url = "https://96xuaw9m48.execute-api.us-east-2.amazonaws.com/test2/transaction?topic="
    payload = {}
    headers = {
        'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAAP7YTAEAAAAAE4h6IXYWi%2BGMp9HwZjpkzxmVzyQ%3D9t7woenbmjW9ACLilj4zh4Vd7o1acUPwjJ10zL46ebYS9EZKGT'
    }
    if not finish and not activated:
        if first:
            first = False
            return "Please, say a gretting to activate Wikibot"
        keyboard = preprocessing(pd.Series(data={'Sentence': keyboard}))
        keyboard = keyboard['Sentence']
        intent, keyword = Intent_detection(keyboard, df)
        if intent == "Greeting":
            activated = True
            return "Hi, this is Wikibot, an NLP based chatbot aimed to help in your wikipedia search.\n " \
                   "You can interact with Wikibot in the following ways:\n" \
                   "1.- Tell me what to search. Use sentences as: search for X, tell me what you know about X, what wikipedia says about X...\n" \
                   "2.- Tell me if you want to get suggestions related to a word. " \
                   "Use sentences as: give suggestions for X, what can you tell me related to X, tell me things connected to X \n" \
                   "3.- Tell me if you want to deactivate Wikibot. Use sentences as: Bye, Have a nice day, Goodbye \n" \
                   "4.- Greet me to reactivate Wikibot. Use sentences as: Hi, how are you?, what\'s up? \n" \
                   "What can I do for you?"
        else:
            return "That is not a greeting, try again if you want to activate Wikibot"

    if activated:

        keyboard = preprocessing(pd.Series(data={'Sentence': keyboard}))
        keyboard = keyboard['Sentence']
        intent, keyword = Intent_detection(keyboard, df)
        if intent == "Greeting":
            return "Wikibot is already activated, try to ask me a question!"
        elif intent == "Search":
            url += keyword
            response = requests.request("GET", url, headers=headers, data=payload)
            if response.status_code == 200:
                return response.text
            else:
                print("\nI have not found any coincidence. \n")
                if wiki.suggest(keyword) is not None:
                    return "Did you mean {} ?\n".format(wiki.suggest(keyword))
                else:
                    return "Try again"
        elif intent == "Suggestions":
            return "This is what Wikibot has found related to {} \n: {}".format(keyword,wiki.suggest(keyword))

        elif intent == "Farewell":
            activated = False
            return "It was a pleasure helping you. Wikibot is now going to sleep!"


def deactivate():
    global activated
    activated == False
