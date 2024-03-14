
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
# global text
# global uploaded_files

os.environ["OPENAI_API_KEY"] = "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"

#upload a file
import streamlit as st
from io import StringIO
import pdfplumber
import time
import fitz   
import re


st.subheader("Step 1:")
name = st.text_input('Name your assistant')


def uploaded_text():
    uploaded_files = st.file_uploader("Upload file(s) for your Knowledge Base (in PDF format)", type='pdf', accept_multiple_files=True)
    with st.spinner('Uploading Knowledge base..'):
            time.sleep(2)
    for uploaded_file in uploaded_files:        
        with pdfplumber.open(uploaded_file) as file:
            all_pages = file.pages
            for i in all_pages:
                uploaded_text.text = i.extract_text_simple()
                # return text
                texts = re.findall(r'\b\w+\b', uploaded_text.text)
                return texts


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


t= uploaded_text()
print(type(t))
for i in t:
    encoded_text = [tokenizer(i)]
print(encoded_text)
def count_tokens(text: str) -> int:
    return len(encoded_text)

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)
chunks = text_splitter.create_documents(encoded_text)
print(chunks)
# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
# type(chunks[0])


# Get embedding model
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(chunks, embeddings)

# Test block to search from kb
query = "I want to return my product"
docs = db.similarity_search(query)

# r = chain.run(input_documents=docs, question=query)
# print("-------------", r)
model_name="gpt-3.5-turbo"
llm=OpenAI(model_name=model_name)
chain = load_qa_chain(llm, chain_type="stuff")
def search_from_kb(q):
    
    docs = db.similarity_search(q)
    op = chain.run(input_documents=docs, question=q)
    return op
def chatbot_response(user_query):
    # Combine the user query with the extracted PDF text for context.
    conversation = uploaded_text.text + "\nUser: " + user_query
    
    # Use GPT-3.4 turbo to generate a response.
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": conversation},
        ],
        max_tokens=100,  # Adjust the response length as needed.
    )
    print("Line 209 -------")
    # print(response)
    return response.choices[0].message.content
# messages=[
#             {"role": "system", "content": "You are a helpful customer support assistant."}
#         ]
def query_refiner(conversation, query):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    # messages=messages,
    messages=[{ "role":"system", "content": f"Given the following user query and conversation log, provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}],
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    output= response.choices[0].message.content
    # print("Line 183 -- ", response.choices[0].message.content)
    return output


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm=ChatOpenAI(model_name="gpt-3.5-turbo",
               openai_api_key="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")

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state['buffer_memory'], prompt=prompt_template, llm=llm, verbose=True)

st.title(name)
# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            # st.subheader("Refined Query:")
            st.write(refined_query)
            context = findmatch(refined_query, "text/plain", t)
            # context = findmatch(refined_query, "text/plain","TemplatesusedinEmails.txt")
            # print("-----",context)  
            resp = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            print("-----",resp)  
            response = chatbot_response(query)
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
 
          

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")

# if query:
#     with st.spinner("typing..."):
#         ...
#         response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
#     st.session_state.requests.append(query)
#     st.session_state.responses.append(response)
# # Send button
#     send_message = st.button("Send")

# from langchain.chat_models import ChatOpenAI
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain.chains import RetrievalQA
# # chat completion llm
# llm = ChatOpenAI(
#     openai_api_key='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',
#     model_name='gpt-3.5-turbo',
#     temperature=0.0
# )
# # conversational memory
# conversational_memory = ConversationBufferWindowMemory(
#     memory_key='chat_history',
#     k=5,
#     return_messages=True
# )




# from IPython.display import display
# import ipywidgets as widgets



# model_name="gpt-3.5-turbo"
# llm=OpenAI(model_name=model_name)

# # Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
# qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1,model_name=model_name), db.as_retriever())

# # chat_history = []

# # def on_submit(_):
# #     query = input_box.value
# #     input_box.value = ""

# #     if query.lower() == 'exit':
# #         print("Thank you for using the IV chatbot!")
# #         return

# #     result = qa({"question": query, "chat_history": chat_history})
# #     chat_history.append((query, result['answer']))

# #     display(widgets.HTML(f'<b>User:</b> {query}'))
# #     display(widgets.HTML(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}'))

# # print("Welcome to the IV chatbot! Type 'exit' to stop.")

# # input_box = widgets.Text(placeholder='Please enter your question:')
# # input_box.on_submit(on_submit)

# # display(input_box)

# # # %%
# # import streamlit as st
# # from streamlit_chat import message

# # if "openai_model" not in st.session_state:
# #     st.session_state["openai_model"]="gpt-3.5-turbo"

# # if "messages" not in st.session_state:
# #     st.session_state.message=[]

# # for message in st.session_state.messages:
# #     with st.chat_message(message["role"]):
# #         st.markdown(message["content"])

# # if prompt:= st.chat_input("Ask something.."):
# #     st.session_state.message.append({"role":"user", "content":prompt})

# # with st.chat_message("user"):
# #     st.markdown(prompt)

# # with st.chat_message("assistant"):
# #     stream = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1,model_name=model_name), db.as_retriever(), stream=True)
# # response= st.write_stream(stream)
# # st.session_state.message.append({"role": "assistant", "content": response})

# import streamlit as st


# chat_history = []

# def get_bot_response(user_input):
#     result = qa({"question": query, "chat_history": chat_history})
#     chat_history.append((query, result['answer']))
#     return f"Echo: {user_input}"

# def chatbot_app():
#     st.title('Simple Chatbot with Conversational Memory')
    
#     # Initialize conversation history if it doesn't exist
#     if 'chat_history' not in st.session_state:
#         st.session_state['chat_history'] = []
    
#     # User input
#     user_input = st.text_input("You: ", key="user_input_field", on_change=clear_input)

#     # Send button
#     send_message = st.button("Send")

#     if send_message:
#         # Prevent adding empty messages
#         if user_input.strip() != "":
#             # Simulate bot response
#             bot_response = get_bot_response(user_input)
            
#             # Update conversation history
#             st.session_state.chat_history.append(f"You: {user_input}")
#             st.session_state.chat_history.append(bot_response)
        
#     # Display conversation
#     if 'chat_history' in st.session_state:
#         for message in st.session_state.chat_history:
#             st.text(message)

# def clear_input():
#     # Clear the input field
#     st.session_state.user_input_field = ""

# if __name__ == "__main__":
#     chatbot_app()
