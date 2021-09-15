
import requests
import validators
import streamlit as st
import pandas as pd
import chatbot_main_modified_2_0

chatbot_main_modified_2_0.deactivate()

st.title("Welcome to SmartBot")

st.markdown(
    "Welcome! With this bot you can get information about whatever you want :smile:"
)
from PIL import Image
image = Image.open('bot.jpg')

st.image(image)

st.write("Please, say a gretting to activate Wikibot")

text = st.text_input("Enter your text", value="")

st.write("Probando "+ text)

lines = chatbot_main_modified_2_0.generate_text(text).split("\n")
for line in lines:
    st.write(line)
