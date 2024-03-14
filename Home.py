
import langchain
import pip
import tiktoken
import transformers
import textract
import openai
import tiktoken
import faiss

# pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu

from mailcap import findmatch
import os
import openai
import pandas as pd
import PyPDF2
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import streamlit as st
from streamlit_chat import message
from utils import *
from langchain.chains import RetrievalQA

# %%
os.environ["OPENAI_API_KEY"] = "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"

#upload a file
import streamlit as st
from io import StringIO
import pdfplumber
import time

import os
path = os.path.abspath("pages/Main.py")

st.title(":blue[ Welcome to IV Chatbot Creator!]")
st.subheader("Create and test your assistants for free!", divider='blue')
st.text("")
st.page_link(path, label= " + Create a New Assistant", icon="➕")

import os
print(os.path.abspath("pages/Main.py"))
