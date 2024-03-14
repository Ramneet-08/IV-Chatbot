import openai
import spacy
import fitz

# Set your OpenAI API key and the GPT-3.4 turbo engine (ensure you have access).
openai.api_key = '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'
engine = 'gpt-3.5-turbo-1106'

# Define a function to extract text from a PDF file.

spacy.cli.download("en_core_web_sm")
def convert_pdf_to_text(pdf_file):
    try:
        text = ""
        pdf_document = fitz.open(pdf_file)
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            text += page.get_text()

        return text
    except Exception as e:
        print(f"Failed to convert the PDF to text. Error: {str(e)}")
        return None

# Specify the path to your PDF file
pdf_file_path = 'TemplatesusedinEmails.pdf'

# Convert the PDF to text
text = convert_pdf_to_text(pdf_file_path)

if text:
    print(text)
else:
    print("Failed to convert the PDF to text.")
# Define a function to answer questions using GPT-3.4 turbo.
def chatbot_response(user_query, pdf_text):
    # Combine the user query with the extracted PDF text for context.
    conversation = pdf_text + "\nUser: " + user_query
    
    # Use GPT-3.4 turbo to generate a response.
    response = openai.ChatCompletion.create(
        model=engine,
        messages=[
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": conversation},
        ],
        max_tokens=100,  # Adjust the response length as needed.
    )
   
    return response.choices[0].text

# # Extract text from the PDF file.
pdf_text = convert_pdf_to_text(pdf_file_path)

# Initialize spaCy NLP for more advanced natural language processing.
nlp = spacy.load('en_core_web_sm')

# Start a conversation with the chatbot.
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == 'exit':
#         break
#     # Process the user query using spaCy for more advanced NLP.
#     doc = nlp(user_input)
#     user_query = ' '.join([token.lemma_ for token in doc])
#     response = chatbot_response(user_query, pdf_text)
#     print("Chatbot:", response)
def main():
    # print("Chatbot: Hi! I'm your chatbot assistant.")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_input, pdf_text)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()











    from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA

# chat completion llm
llm = ChatOpenAI(
    openai_api_key='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',
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)