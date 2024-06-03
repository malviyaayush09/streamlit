import json
import os
import asyncio
import streamlit as st
import openai
from tempfile import NamedTemporaryFile
import pymupdf
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index.core import Document, ServiceContext, set_global_service_context
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

os.environ['OPENAI_API_KEY'] = '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'

def get_or_create_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop = asyncio.get_event_loop()
    return loop

@st.cache_resource
def create_datastax_connection():
    cloud_config = {'secure_connect_bundle': r"C:\Users\AMALVIYA\Downloads\nlp\nlp\secure-connect-nlp.zip"}
    with open(r"C:\Users\AMALVIYA\Downloads\nlp\nlp\premkumarc1111@gmail.com-token.json") as f:
        secrets = json.load(f)
    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]
    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    astra_session = cluster.connect()
    return astra_session

def read_pdf(file_path):
    text = ""
    with pymupdf.open(file_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text

def main():
    index_placeholder = None
    st.set_page_config(page_title="Chat with your PDF using OpenAI GPT model", page_icon="📄")
    st.header('Chat with your PDF using OpenAI GPT model')
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message['avatar']):
            st.markdown(message["content"])
    
    session = create_datastax_connection()
    
    openai.api_key = os.getenv("OPENAI_API_KEY")

    loop = get_or_create_event_loop()
    
    # You can set parameters for OpenAI GPT model according to your requirement
    # For example, set temperature, max tokens, etc.
    response_mode = "temperature"  # or "best_of"
    temperature = 0.7
    max_tokens = 150
    
    with st.sidebar:
        st.subheader('Upload Your PDF File')
        docs = st.file_uploader('⬆️ Upload your PDF & Click to process',
                                accept_multiple_files=False,
                                type=['pdf'])
        if st.button('Process'):
            if docs is not None:
                try:
                    with NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        temp_file.write(docs.getbuffer())
                        temp_file_path = temp_file.name
                    
                    with st.spinner('Processing'):
                        pdf_text = read_pdf(temp_file_path)
                        os.remove(temp_file_path)  # Clean up temp file
                        
                        document = Document(text=pdf_text)
                        documents = [document]
                        index = VectorStoreIndex.from_documents(documents, service_context=None)
                        query_engine = index.as_query_engine()
                        if "query_engine" not in st.session_state:
                            st.session_state.query_engine = query_engine
                        st.session_state.activate_chat = True
                except Exception as e:
                    st.error(f"Error processing the file: {e}")
    
    if st.session_state.activate_chat:
        if prompt := st.chat_input("Ask your question from the PDF?"):
            with st.chat_message("user", avatar='👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "avatar": '👨🏻', "content": prompt})
            
            query_index_placeholder = st.session_state.query_engine
            pdf_response = query_index_placeholder.query(prompt)
            
            # Using OpenAI GPT model to generate response
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=pdf_response.response,
                response_mode=response_mode,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            cleaned_response = response.choices[0].text.strip()
            
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "assistant", "avatar": '🤖', "content": cleaned_response})
        else:
            st.markdown('Upload your PDFs to chat')

if __name__ == '__main__':
    main()
