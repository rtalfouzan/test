import streamlit as st
from GoogleNews import GoogleNews

import pandas as pd
import numpy as np
import spacy
import gensim
import string
import re

import sklearn
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("spacy.aravec.model")


st.write("""
testing testing
""")


text_input = st.text_input (''' **ادخل the text** ''')


st.sidebar.markdown('مواقع اخباريه معتمده ')
st.sidebar.markdown("[العربية](https://www.alarabiya.net/)")
st.sidebar.markdown("[الجزيرة نت](https://www.aljazeera.net/news/)")
st.sidebar.markdown("[وكالة الانباء الكويتية](https://www.kuna.net.kw/Default.aspx?language=ar)")
