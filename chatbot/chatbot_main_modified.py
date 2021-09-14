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

from  Intent_detection_2_0 import Intent_detection_function
import wikipedia as wiki

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
finish = False
activated = False

df = pd.read_excel('database_intents.xlsx')

def generate_text(keyboard):
    global finish 
    global activated
    print(1)
    url = "https://96xuaw9m48.execute-api.us-east-2.amazonaws.com/test2/transaction?topic="
    payload = {}
    headers = {
        'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAAP7YTAEAAAAAE4h6IXYWi%2BGMp9HwZjpkzxmVzyQ%3D9t7woenbmjW9ACLilj4zh4Vd7o1acUPwjJ10zL46ebYS9EZKGT'
    }
    if not finish and not activated:
        intent, keyword = Intent_detection_function(keyboard, df)
        if intent == "Greeting":
            activated = True
            return "Hi, this is Wikibot, an NLP based chatbot aimed to help in your wikipedia search.\n What can I do for you?"
    if activated:
        intent, keyword = Intent_detection_function(keyboard, df)
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
                    return "Did you mean {} \n".format(wiki.suggest(keyword))
                else:
                    return "Try again"
        elif intent == "Suggestions":
            return "This is what Wikibot has found related to {} \n: {}".format(keyword)+"{}".format((wiki.suggest(keyword)))

        elif intent == "Farewell":
            activated = False
            return "It was a pleasure helping you. Wikibot is now going to sleep!"
    if not activated:
        return "That is not a greeting, try again if you want to activate Wikibot"


def deactivate():
    global activated
    activated == False
