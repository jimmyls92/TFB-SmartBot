import random

from Intent_detection_3_0 import Intent_detection

random.seed(42)
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import wikipedia as wiki
import re

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


finish = False
activated = False
while finish == False:

    if activated == False:
        print("Please, say a gretting to activate Wikibot")
        keyboard = input()

        keyboard = preprocessing(pd.Series(data={'Sentence': keyboard}))
        keyboard = keyboard['Sentence']
        print(keyboard)
        intent, keyword = Intent_detection(keyboard, df)

        if intent == "Greeting":
            activated = True
            print("Hi, this is Wikibot, an NLP based chatbot aimed to help in your wikipedia search")
            print('You can interact with Wikibot in the following ways:')
            print('\n 1.- Tell me what to search. Use sentences as: search for X, tell me what you know about X, what wikipedia says about X...')
            print('\n 2.- Tell me if you want to get suggestions related to a word.')
            print('Use sentences as: give suggestions for X, what can you tell me related to X, tell me things connected to X')
            print('\n 3.- Tell me if you want to deactivate Wikibot. Use sentences as: Bye, Have a nice day, Goodbye')
            print('\n 4.- Greet me to reactivate Wikibot. Use sentences as: Hi, how are you?, what\'s up?')

        else:
            print("That is not a greeting, try again if you want to activate Wikibot")
            continue

    if activated == True:

        while activated == True:
            print("\n What can I do for you?")
            keyboard = input()
            keyboard = preprocessing(pd.Series(data={'Sentence': keyboard}))
            keyboard = keyboard['Sentence']
            print(keyboard)
            intent, keyword = Intent_detection(keyboard, df)

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










