import random 

import joblib

import sys

import random

from Intent_detection_embbeded import Intent_detection_embbeded

random.seed(42)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pickle
import warnings
warnings.filterwarnings("ignore")
import wikipedia as wiki

df = pd.read_excel('database_intents.xlsx', engine='openpyxl')


finish = False
activated = False
while finish == False:

    if activated == False:
        print("Please, say a gretting to activate Wikibot")
        keyboard = input()
        intent, keyword = Intent_detection_embbeded(keyboard, df)

        if intent == "Greeting":
            activated = True
            print("Hi, this is Wikibot, an NLP based chatbot aimed to help in your wikipedia search")
        else:
            print("That is not a greeting, try again if you want to activate Wikibot")
            continue

    if activated == True:

        while activated == True:
            print("What can I do for you?")
            keyboard = input()
            intent, keyword = Intent_detection_embbeded(keyboard, df)

            if intent == "Greeting":
                print("Wikibot is already activated, try to ask me a question!")

            elif intent == "Search":
                try:
                    #print("He encontrado: .{}".format(wiki.search(keyboard, results)))
                    print(wiki.summary(keyword))
                except:
                    print("\nI have not found any coincidence. \n")
                    if wiki.suggest(keyword) != None:
                        print("Did you mean {} \n".format(wiki.suggest(keyword)))
                    else:
                        print("Try again")

            elif intent == "Suggestions":
                print("This is what Wikibot has found related to {}:".format(keyword))
                print("{}".format((wiki.suggest(keyword))))

            elif intent == "Farewell":
                print("It was a pleasure helping you. Wikibot is now going to sleep!")
                activated = False










