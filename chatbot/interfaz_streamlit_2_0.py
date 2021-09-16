
import requests
import validators
import streamlit as st
import pandas as pd
import chatbot_main_cloud

chatbot_main_cloud.deactivate()

st.title("Welcome to SmartBot")

st.markdown(
    "Welcome! With this bot you can get information about whatever you want :smile:"
)
from PIL import Image
image = Image.open('bot.jpg')

st.image(image)

st.write("")

text = st.text_input("Enter your text", value="")

lines = chatbot_main_cloud.generate_text(text).split("\n")
for line in lines:
    st.write(line)
