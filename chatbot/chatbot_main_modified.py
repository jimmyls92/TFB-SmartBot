from Intent_detection_function import Intent_detection_function
import wikipedia as wiki

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
finish = False
activated = False

def generate_text(keyboard):
    global finish 
    global activated 
    if finish == False and activated == False:
        intent, keyword = Intent_detection_function(keyboard)
        if intent == "Greeting":
            activated = True
            return "Hi, this is Wikibot, an NLP based chatbot aimed to help in your wikipedia search.\n What can I do for you?"
    if activated == True:
        intent, keyword = Intent_detection_function(keyboard)
        if intent == "Greeting":
            return "Wikibot is already activated, try to ask me a question!"
        elif intent == "Search":
            try:
                # print("He encontrado: .{}".format(wiki.search(keyboard, results)))
                return wiki.summary(keyword)
            except:
                print("\nI have not found any coincidence. \n")
                if wiki.suggest(keyword) != None:
                    return "Did you mean {} \n".format(wiki.suggest(keyword))
                else:
                    return "Try again"

        elif intent == "Suggestions":
            return "This is what Wikibot has found related to {} \n: {}".format(keyword)+"{}".format((wiki.suggest(keyword)))

        elif intent == "Farewell":
            activated = False
            return "It was a pleasure helping you. Wikibot is now going to sleep!"
    if activated == False:
        return "That is not a greeting, try again if you want to activate Wikibot"
        
def deactivate():
    activated == False
