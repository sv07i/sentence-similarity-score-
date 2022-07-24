import streamlit as st
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer, util

from sklearn.metrics.pairwise import cosine_similarity
pipe = pickle.load(open('model.pkl','rb'))

similarity = pickle.load(open('similarity.pkl','rb'))


st.title("similarity ")


sentence_1 = st.text_input("Enter Your First Sentence ")

sentence_2 = st.text_input("Enter Your Second Sentence ")



_model = 'sentence-transformers/bert-base-nli-mean-tokens'
model = SentenceTransformer(_model)

embeding1  = model.encode(sentence_1, convert_to_tensor=True)

embeding2 = model.encode(sentence_2, convert_to_tensor=True)

st.button('Predict Similarity')
similarity = (util.pytorch_cos_sim(embeding1, embeding2))


st.title("ABS Similarity " + str(similarity))




st.text(" *Empty sentence show similarity of 1*")

