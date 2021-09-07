
import requests
import validators
import streamlit as st
import pandas as pd
import chatbot_main_modified


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

st.write(chatbot_main_modified.generate_text(text))
